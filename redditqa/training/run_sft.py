import argparse
import os

from accelerate import Accelerator
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed,
)
from trl import SFTTrainer

import wandb
from redditqa.data import qa_generation
from redditqa.data.continuous_learning import add_continuous_learning_dataset
from redditqa.data.loader import load_dataset

# Set up logging to only show errors
logging.set_verbosity_info()


# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_freq", default=5, type=int)

    parser.add_argument("--sanity_check", action="store_true", default=False)

    parser.add_argument("--continuous_learning_subset", type=int, default=1000)

    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    # Get args
    args = get_args()

    # Setup WandB
    wandb.init(entity="reddit-qa", project="reddit-qa-paper-eli5", name=os.path.basename(args.output_dir))
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    # Fix the seed for reproducibility
    set_seed(args.seed)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset as pairs of questions and best answers
    dataset = load_dataset(name="askhistorians", task="sft")
    if args.continuous_learning_subset:
        dataset = add_continuous_learning_dataset(
            dataset,
            task="sft",
            subset=args.continuous_learning_subset,
            tokenizer=AutoTokenizer.from_pretrained(args.model_path),
        )
    print("Has dataset")
    print(dataset)

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

    run_name = args.output_dir.split("/")[-1]
    training_args = TrainingArguments(
        # Epochs
        evaluation_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.log_freq,
        # Batch size
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # LR
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # Other
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=args.bf16,
        # Output
        run_name=run_name,
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
    print_trainable_parameters(trainer.model)
    print(f"Training for {args.num_train_epochs} epochs")
    trainer.train()

    # Save the model
    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


if __name__ == "__main__":
    main()
