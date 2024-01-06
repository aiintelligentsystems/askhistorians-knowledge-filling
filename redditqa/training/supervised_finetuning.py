import argparse
import os
from functools import partial

import datasets as ds
from accelerate import Accelerator
from huggingface_hub import login
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
    logging,
    set_seed,
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from redditqa.data import askhistorians, ultrachat

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)

DATASETS_CACHE_DIR_PATH = "/scratch1/redditqa/cached_datasets"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--orig_dataset_subset", default=1000, type=int)

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

def create_datasets(tokenizer, args):
    ultrachat_ds = ultrachat.prepare_dataset(tokenizer, subset=args.orig_dataset_subset)
    askhistorians_ds = askhistorians.load_dataset(split=True, task='sft')
    
    train_data = ds.concatenate_datasets([ultrachat_ds['train'], askhistorians_ds['train']]).shuffle(seed=42)
    valid_data = ds.concatenate_datasets([ultrachat_ds['eval'], askhistorians_ds['eval']]).shuffle(seed=42)

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        dataset_text_field="text",
        infinite=True,
        seq_length=args.seq_length,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        dataset_text_field="text",
        infinite=False,
        seq_length=args.seq_length,
    )

    return train_dataset, valid_dataset

def run_training(args, train_data, val_data, tokenizer=None):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    run_name = args.output_dir.split("/")[-1]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=run_name,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    if "llama-2" in args.model_path.lower():
        assert tokenizer is not None, "Please provide a tokenizer for LLama"

        model = LlamaForCausalLM.from_pretrained(
            args.model_path, load_in_4bit=True, device_map={"": Accelerator().process_index}
        )
        model.resize_token_embeddings(model.config.vocab_size + 1)
        model.config.update(dict(pad_token_id=tokenizer.pad_token_id))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    config = AutoConfig.from_pretrained(args.model_path)
    architecture = config.architectures[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if "llama-2" in architecture.lower():
        print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
