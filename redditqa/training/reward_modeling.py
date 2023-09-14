import os
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import wandb
from redditqa.dataset import load_reddit_dataset
from trl import RewardConfig, RewardTrainer

# Set up wandb
wandb.init(
    project="reward-modeling",
)

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)


# Fix the seed for reproducibility
SEED = 42
set_seed(SEED)


@dataclass
class ScriptArguments:
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
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
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    output_dir: Optional[str] = field(default="")
    eval_subsample: Optional[int] = field(default=5000)
    eval_steps: Optional[int] = field(default=1000)
    margin_mode: Optional[str] = field(default=None, metadata={"help": "The margin mode to use."})


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def build_dataset(tokenizer, max_length, eval_subsample, margin_mode):
    # Load the reddit dataset for tuning the reward model.
    train_dataset = load_reddit_dataset("train", pairs=True)
    eval_dataset = load_reddit_dataset("eval", pairs=True)

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
            "margin": [],
        }
        for question_title, response_j, response_k, score_j, score_k in zip(
            examples["question_title"],
            examples["response_j"],
            examples["response_k"],
            examples["score_j"],
            examples["score_k"],
        ):
            template = "<|ELIF|> Question: %question\nAnswer: %answer"

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

            if margin_mode:
                margin = score_j - score_k
                if margin_mode == "inverse":
                    margin *= -1
            new_examples["margin"].append(margin)

        return new_examples

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    num_proc = 1  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        preprocess_function, num_proc=num_proc, remove_columns=original_columns, batched=True
    )
    eval_dataset = eval_dataset.map(
        preprocess_function, num_proc=num_proc, remove_columns=original_columns, batched=True
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(eval_subsample))

    print("Finished preprocessing dataset.")
    print("Number of training examples: ", len(train_dataset))
    print("Number of eval examples: ", len(eval_dataset))

    return train_dataset, eval_dataset


def load_model(model_name, tokenizer, gradient_checkpointing):
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
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16
        # device_map="auto",
    )
    # if "llama-2" in script_args.model_name.lower():
    #     assert tokenizer is not None, "Please provide a tokenizer for LLama"
    #
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         script_args.model_name,
    #         num_labels=1,
    #         load_in_8bit=True,
    #         device_map="auto",
    #     )
    #     model.resize_token_embeddings(model.config.vocab_size + 1)
    #     model.config.update(dict(pad_token_id=tokenizer.pad_token_id))
    # else:
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         script_args.model_name,
    #         num_labels=1,
    #         load_in_8bit=True,
    #         device_map="auto",
    #     )
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not gradient_checkpointing
    model = model.cuda()

    return model


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    print("Using output output_dir: ", script_args.output_dir)
    print("Eval steps: ", script_args.eval_steps)
    training_args = RewardConfig(
        output_dir=script_args.output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.eval_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        report_to="wandb",
        max_length=script_args.max_length,
    )

    model = load_model(script_args.model_name, tokenizer, script_args.gradient_checkpointing)
    train_dataset, eval_dataset = build_dataset(
        tokenizer, script_args.max_length, script_args.eval_subsample, script_args.margin_mode
    )

    # Train the model
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(script_args.output_dir + "_peft_last_checkpoint")
    trainer.evaluate()


if __name__ == "__main__":
    main()
