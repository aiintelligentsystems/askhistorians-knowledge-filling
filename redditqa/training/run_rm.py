import os
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import login
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    set_seed,
)
from trl import RewardConfig, RewardTrainer

import wandb
from redditqa.data import pair_generation
from redditqa.data.load_eli5 import load_eli5

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)


# Fix the seed for reproducibility
SEED = 42
set_seed(SEED)


@dataclass
class ScriptArguments:
    batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="",
        metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    output_dir: Optional[str] = field(default="")
    eval_subsample: Optional[int] = field(default=5000)
    eval_steps: Optional[int] = field(default=1000)
    score_margin: Optional[str] = field(
        default=None, metadata={"help": "Consider only pairs with that score margin or above"}
    )


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            control.should_evaluate = True


def build_dataset(tokenizer, max_length, eval_subsample, score_margin=None):
    # Load the reddit dataset for tuning the reward model.
    dataset = load_eli5()
    dataset = pair_generation.apply(dataset, score_margin=score_margin)
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
        for question_title, response_j, response_k, score_j, score_k in zip(
            examples["question_title"],
            examples["response_j"],
            examples["response_k"],
            examples["score_j"],
            examples["score_k"],
        ):
            template = "Question: %question\nAnswer: %answer"

            text_j = template.replace("%question", question_title).replace("%answer", response_j)
            text_k = template.replace("%question", question_title).replace("%answer", response_k)
            tokenized_j = tokenizer(text_j, truncation=True)
            tokenized_k = tokenizer(text_k, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
            new_examples["score_chosen"].append(score_j)
            new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])
            new_examples["score_rejected"].append(score_k)

        return new_examples

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
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
    print(f"Size before filtering to max_length={max_length}: train={len(train_dataset)}, eval={len(eval_dataset)}")
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    print(f"Size after filtering to max_length={max_length}: train={len(train_dataset)}, eval={len(eval_dataset)}")
    eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(eval_subsample))

    print("Finished preprocessing dataset.")
    print("Number of training examples: ", len(train_dataset))
    print("Number of eval examples: ", len(eval_dataset))

    return train_dataset, eval_dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print("Using output output_dir: ", script_args.output_dir)
    assert script_args.output_dir, "You must specify an output_dir."

    # Setup WandB
    wandb.init(entity="reddit-qa", project="reddit-qa-paper-eli5", name=os.path.basename(script_args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Lora config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        load_in_8bit=True,
        device_map="auto",
    )
    if not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.eos_token_id

    training_args = RewardConfig(
        output_dir=script_args.output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.eval_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type=script_args.lr_scheduler_type,
        run_name=script_args.output_dir.split("/")[-1],
        report_to="wandb",
        max_length=script_args.max_length,
    )

    train_dataset, eval_dataset = build_dataset(
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        eval_subsample=script_args.eval_subsample,
        score_margin=script_args.score_margin,
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

    # Add a callback to evaluate the model after the first step
    trainer.add_callback(EvaluateFirstStepCallback())

    # Train
    trainer.train()

    # Save the final checkpoint
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
