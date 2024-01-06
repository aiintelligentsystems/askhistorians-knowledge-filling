import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import login
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import RewardConfig, RewardTrainer

import wandb
from redditqa.data import pair_generation
from redditqa.data.load_eli5 import load_eli5
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

    num_train_epochs: Optional[int] = field(default=5)
    eval_steps: Optional[int] = field(default=1000)

    learning_rate: Optional[float] = field(default=2e-5)
    lr_scheduler_type: Optional[str] = field(default="linear")

    batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    max_length: Optional[int] = field(default=512)

    score_margin: Optional[str] = field(
        default=None, metadata={"help": "Consider only pairs with that score margin or above"}
    )


def build_dataset(dataset_name, tokenizer, max_length, eval_subsample, score_margin=None):
    # Load the reddit dataset for tuning the reward model.
    dataset = load_dataset(name=dataset_name, task="dpo", eval_subsample=eval_subsample, score_margin=score_margin)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    # Then tokenize the dataset.
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "score_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "score_rejected": [],
        }
        for prompt, chosen, rejected, score_choosen, score_rejected in zip(
            examples["prompt"],
            examples["chosen"],
            examples["rejected"],
            examples["choosen_score"],
            examples["rejected_score"],
        ):
            tokenized_chosen = tokenizer(prompt + chosen, truncation=True)
            tokenized_rejected = tokenizer(prompt + rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["score_chosen"].append(score_choosen)
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
            new_examples["score_rejected"].append(score_rejected)

        return new_examples

    # Preprocess the dataset
    num_proc = 1  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        preprocess_function,
        num_proc=num_proc,
        remove_columns=original_columns,
        batched=True,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        num_proc=num_proc,
        remove_columns=original_columns,
        batched=True,
    )

    # Filter out examples that are too long
    print(f"Size before filtering to max_length={max_length}: train={len(train_dataset)}, eval={len(eval_dataset)}")
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    print(f"Size after filtering to max_length={max_length}: train={len(train_dataset)}, eval={len(eval_dataset)}")
    eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(eval_subsample))

    # Print size of the dataset
    print("Finished preprocessing dataset.")
    print("Number of training examples: ", len(train_dataset))
    print("Number of eval examples: ", len(eval_dataset))

    return train_dataset, eval_dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup WandB
    wandb.init(entity="reddit-qa", project=args.wandb_project, name=os.path.basename(args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    train_dataset, eval_dataset = build_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        eval_subsample=args.eval_subsample,
        score_margin=args.score_margin,
    )

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        load_in_8bit=True,
        device_map="auto",
    )

    # Load the Lora config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = RewardConfig(
        # Training epochs
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
        # Output
        run_name=os.path.basename(args.output_dir),
        output_dir=args.output_dir,
        report_to="wandb",
        # Other
        bf16=True,
        remove_unused_columns=False,
        label_names=[],
        # Max length
        max_length=args.max_length,
    )

    # Train the model
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    trainer.train()

    # Save the final checkpoint
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
