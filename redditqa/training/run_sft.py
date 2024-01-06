import os
from dataclasses import dataclass, field
from typing import Optional

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
from trl import SFTTrainer

import wandb
from redditqa.data.continuous_learning import add_continuous_learning_dataset
from redditqa.data.loader import load_dataset

# Set up logging to show full logs
logging.set_verbosity_info()

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)

# Fix the seed for reproducibility
SEED = 42
set_seed(SEED)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field()
    output_dir: Optional[str] = field()
    wandb_project: Optional[str] = field()

    num_train_epochs: Optional[int] = field(default=1)
    eval_steps: Optional[int] = field(default=100)
    eval_subsample: Optional[int] = field(default=1000, metadata={"help": "the evaluation subsample"})

    learning_rate: Optional[float] = field(default=1e-5)
    lr_scheduler_type: Optional[str] = field(default="cosine")

    batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    max_seq_length: Optional[int] = field(default=1024)

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})

    dataset_name: Optional[str] = field()
    continuous_learning_subset: Optional[int] = field(
        default=1000, metadata={"help": "original dataset subset used for continual learning rehearsal"}
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup WandB
    wandb.init(entity="reddit-qa", project=args.wandb_project, name=os.path.basename(args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset as pairs of questions and best answers
    dataset = load_dataset(name=args.dataset_name, task="sft", eval_subsample=args.eval_subsample)
    if args.continuous_learning_subset:
        dataset = add_continuous_learning_dataset(
            dataset,
            task="sft",
            subset=args.continuous_learning_subset,
            tokenizer=AutoTokenizer.from_pretrained(args.model_path),
        )
    print("Has dataset")
    print(dataset)
    # Print the average length of packed sequences
    train_lengths = [len(x["full_text"]) for x in dataset["train"]]
    average_length = sum(train_lengths) / len(train_lengths)
    print(f"Average length per packed sequence: {average_length / args.max_seq_length}")

    # Truncate the dataset for debugging if sanity_check is True
    if args.sanity_check:
        dataset["train"] = dataset["train"].shuffle().select(range(100))
        dataset["eval"] = dataset["eval"].shuffle().select(range(100))
        args.gradient_accumulation_steps = 1

    # Load model
    print("Loading the model")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True, device_map="cuda:0")

    training_args = TrainingArguments(
        # Epochs
        num_train_epochs=args.num_train_epochs,
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
        # Other
        bf16=True,
        # Output
        run_name=os.path.basename(args.output_dir),
        output_dir=args.output_dir,
        report_to="wandb",
        overwrite_output_dir=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="full_text",
        max_seq_length=args.max_seq_length,
        packing=False,
        peft_config=lora_config,
    )

    # Train the model
    print(f"Training for {args.num_train_epochs} epochs")
    wandb.watch(trainer.model)
    trainer.train()

    # Save the model
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
