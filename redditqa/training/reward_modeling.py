import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
import wandb
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy

from redditqa.dataset import load_reddit_dataset

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
        default="adamw_hf",
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


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
# if "llama-2" in script_args.model_name.lower():
#     # LLAMA 2
#     print("Setting pad token for LLama-2 tokenizer")
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
# elif "llama" in script_args.model_name:
#     # LLAMA 1
#     # required for llama
#     DEFAULT_PAD_TOKEN = "[PAD]"
#     DEFAULT_EOS_TOKEN = "</s>"
#     DEFAULT_BOS_TOKEN = "</s>"
#     DEFAULT_UNK_TOKEN = "</s>"
#     tokenizer.add_special_tokens(
#         {
#             "eos_token": DEFAULT_EOS_TOKEN,
#             "bos_token": DEFAULT_BOS_TOKEN,
#             "unk_token": DEFAULT_UNK_TOKEN,
#             "pad_token": DEFAULT_PAD_TOKEN,
#         }
#     )
# else:
# # required for gpt2
# tokenizer.pad_token = tokenizer.eos_token

# Load the reddit dataset for tuning the reward model.
train_dataset = load_reddit_dataset("train", pairs=True)
eval_dataset = load_reddit_dataset("eval", pairs=True)


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question_title, response_j, response_k in zip(
        examples["question_title"], examples["response_j"], examples["response_k"]
    ):
        template = "<|ELIF|> Question: %question\nAnswer: %answer"

        text_j = template.replace("%question", question_title).replace("%answer", response_j)
        text_k = template.replace("%question", question_title).replace("%answer", response_k)
        tokenized_j = tokenizer(text_j, truncation=True)
        tokenized_k = tokenizer(text_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
num_proc = 1  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names
train_dataset = train_dataset.map(preprocess_function, num_proc=num_proc, remove_columns=original_columns, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, num_proc=num_proc, remove_columns=original_columns, batched=True)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(script_args.eval_subsample))

print("Finished preprocessing dataset.")
print("Number of training examples: ", len(train_dataset))
print("Number of eval examples: ", len(eval_dataset))


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding or "max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding or "max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    print(f"Evaluating predictions with shape: {len(eval_pred.predictions.shape)}")
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    result = accuracy.compute(predictions=predictions, references=labels)
    return result


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
print("Using output output_dir: ", script_args.output_dir)
print("Eval steps: ", script_args.eval_steps)
training_args = TrainingArguments(
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
)


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
model.config.use_cache = not script_args.gradient_checkpointing
model.print_trainable_parameters()
model = model.cuda()


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()

print("Saving last checkpoint of the model")
model.save_pretrained(script_args.output_dir + "_peft_last_checkpoint")

trainer.evaluate()
