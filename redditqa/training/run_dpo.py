import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional

import datasets as ds
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer

import wandb
from redditqa.data import pair_generation
from redditqa.data.load_eli5 import load_eli5
from redditqa.data.smart_filter import question_filter

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        metadata={"help": "the location of the SFT model name or path"}, default=""
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    batch_size: Optional[int] = field(default=4, metadata={"help": "batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1e6, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(metadata={"help": "the output directory"}, default="")
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    seed: Optional[int] = field(default=42, metadata={"help": "The seed to use"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})


def get_reddit_dataset_paired(
    sanity_check: bool = False,
    num_proc=1,
) -> Dataset:
    """Load the redditqa dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: %question\nAnswer: "
    """
    # Load the dataset
    dataset = load_eli5()
    dataset = pair_generation.apply(dataset)

    if sanity_check:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["eval"] = dataset["eval"].select(range(100))
        print("Sanity check: only using 100 samples")
        print(dataset)

    def return_prompt_and_responses(row) -> Dict[str, str]:
        prompt_template = "Question: %question\nAnswer: "
        return {
            "prompt": prompt_template.replace("%question", row["question_title"]),
            "chosen": row["response_j"],
            "rejected": row["response_k"],
        }

    dataset = dataset.map(return_prompt_and_responses, num_proc=num_proc)
    return dataset


def main():
    print(f"Has GPU: {torch.cuda.is_available()}")

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # Setup WandB
    wandb.init(project="reddit-qa-paper-eli5", name=os.path.basename(script_args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Load the paired dataset
    dataset = get_reddit_dataset_paired(sanity_check=script_args.sanity_check)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="cuda:0",
    )

    # Create the lora adapter
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Make sure we can train a full batch when sanity checking
    if script_args.sanity_check:
        script_args.gradient_accumulation_steps = 1
        script_args.max_steps = 10

    # Set training args
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to="wandb",
        lr_scheduler_type=script_args.lr_scheduler_type,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=os.path.basename(script_args.output_dir),
    )

    # Initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # Run training
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # Save model
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
