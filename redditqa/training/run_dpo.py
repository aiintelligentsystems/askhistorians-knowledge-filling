import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    logging,
    set_seed,
)
from trl import DPOTrainer

import wandb
from redditqa.data.continuous_learning import add_continuous_learning_dataset
from redditqa.data.loader import load_dataset

# Set up logging to show full logs
logging.set_verbosity_info()

# Fix the seed for reproducibility
SEED = 42
set_seed(SEED)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field()
    output_dir: Optional[str] = field()
    wandb_project: Optional[str] = field()
    dataset_name: Optional[str] = field(default=None)

    max_steps: Optional[int] = field(default=1e6)
    eval_steps: Optional[int] = field(default=100, metadata={"help": "reduce eval set size"})
    eval_subsample: Optional[int] = field(default=None, metadata={"help": "the evaluation subsample"})

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    batch_size: Optional[int] = field(default=4, metadata={"help": "batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=4)
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})

    score_margin: Optional[str] = field(
        default=None, metadata={"help": "Consider only pairs with that score margin or above"}
    )
    continuous_learning_subset: Optional[int] = field(
        default=1000, metadata={"help": "original dataset subset used for continual learning rehearsal"}
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup WandB
    wandb.init(entity="reddit-qa", project=args.wandb_project, name=os.path.basename(args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load the dataset
    dataset = load_dataset(name=args.dataset_name, task="dpo", eval_subsample=args.eval_subsample)
    if args.continuous_learning_subset:
        dataset = add_continuous_learning_dataset(
            dataset,
            task="dpo",
            subset=args.continuous_learning_subset,
            tokenizer=tokenizer,
        )

    # Truncate the dataset for debugging if sanity_check is True
    # Make sure we can train a full batch when sanity checking
    if args.sanity_check:
        dataset["train"] = dataset["train"].shuffle().select(range(100))
        dataset["eval"] = dataset["eval"].shuffle().select(range(100))
        args.gradient_accumulation_steps = 1
        args.max_steps = 10
        args.eval_steps = 5

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        load_in_4bit=True if args.sanity_check else False
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        load_in_4bit=True if args.sanity_check else False
    )

    # Create the lora adapter
    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    # Set training args
    training_args = TrainingArguments(
        # Training steps
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=10,
        # Batch size
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # LR
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer_type,
        # Other
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # Output
        run_name=os.path.basename(args.output_dir),
        output_dir=args.output_dir,
        report_to="wandb",
    )

    # Initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_seq_length,
    )

    # Run training
    wandb.watch(dpo_trainer.model)
    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    # Save model
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
