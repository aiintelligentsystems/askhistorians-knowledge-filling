import gc
import os
from dataclasses import dataclass, field
from typing import Optional

import huggingface_hub
import torch
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from redditqa.data import load_reddit_dataset

# Login to the HuggingFace Hub
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
if HUGGINGFACE_TOKEN is not None:
    login(token=HUGGINGFACE_TOKEN)


tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(metadata={"help": "the model name. GPT2 is not compatible"})
    tokenizer_name: Optional[str] = field(metadata={"help": "tokenizer name"})
    reward_model_name: Optional[str] = field(metadata={"help": "the reward model name"})
    output_dir: Optional[str] = field(default="", metadata={"help": "dir to save the model"})

    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})

    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})

    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )

    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})

    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with="wandb",
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# Create the tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_dataset(tokenizer):
    # We retrieve the dataloader by calling the `build_dataset` function.
    ds = load_reddit_dataset(split="train", pairs=False)
    # train_dataset = train_dataset.select(range(100000))

    original_columns = ds.column_names
    num_proc = 1

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question_title in examples["question_title"]:
            query = f"<|ELIF|> Question: {question_title}\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    ds.set_format(type="torch")
    return ds


dataset = build_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
print(f"Loading model from {config.model_name}")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
model.cuda()


print("Done")


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


print_trainable_parameters(model)


optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

model.config.pad_token_id = model.config.eos_token_id

print(f"Loading reward model from {script_args.reward_model_name}")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_name,
    num_labels=1,
)
reward_model.cuda()
print(reward_model)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    # "batch_size": 16,
    # "truncation": True,
}

# sentiment_pipe.tokenizer.pad_token = sentiment_pipe.tokenizer.eos_token
# sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Maximum epochs to run
    if epoch >= config.total_ppo_epochs:
        break

    # Generate responses
    question_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    print(f"Question: {batch['query']}")
    print(f'Generated responses: {batch["response"]}')

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    print(f"Texts: {texts}")
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
    print(f"Calculated rewards: {rewards}")

    # Free up memory after running the reward model
    torch.cuda.empty_cache()

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    print(f"Epoch {epoch} stats: {stats}")

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        try:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}", push_to_hub=False)
        except huggingface_hub.utils._headers.LocalTokenNotFoundError:
            pass

try:
    ppo_trainer.save_pretrained(script_args.output_dir + f"final", push_to_hub=False)
except huggingface_hub.utils._headers.LocalTokenNotFoundError:
    pass
