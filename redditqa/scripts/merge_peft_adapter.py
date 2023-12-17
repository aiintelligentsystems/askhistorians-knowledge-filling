from dataclasses import dataclass, field
from os.path import join
from typing import Optional

import peft
import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import _get_submodules
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Check the arguments
assert script_args.adapter_model_name is not None, "Please provide the adapter name"
assert script_args.base_model_name is not None, "Please provide the name of the Base model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # peft is for reward model so load sequence classification
    print("Has SEQ_CLS task type")
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the Lora model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

# Merge the adapter
model = model.merge_and_unload()

# Save the model
output_name = script_args.adapter_model_name
output_name = f"{output_name}_merged"
print(f"Saving to {output_name}")
model.save_pretrained(f"{output_name}")
tokenizer.save_pretrained(f"{output_name}")
