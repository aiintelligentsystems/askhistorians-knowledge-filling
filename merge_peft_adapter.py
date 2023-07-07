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
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None)
    checkpoint_dir: Optional[str] = field(default=None)
    base_model_name: Optional[str] = field(default=None)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert (
    script_args.adapter_model_name is not None or script_args.checkpoint_dir is not None
), "please provide an adapter or checkpoint"
assert script_args.base_model_name is not None, "please provide the name of the Base model"


if script_args.checkpoint_dir is not None:
    model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name)

    # This needs to match the training configuration _exactly_.
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # Load the checkpoint
    full_state_dict = torch.load(join(script_args.checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model, full_state_dict)

    model.eval()

elif script_args.adapter_model_name is not None:
    peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
    model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name, return_dict=True)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
    model.eval()


tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)


# For Llama, we need to set the EOS, BOS, and UNK tokens
config = AutoConfig.from_pretrained(script_args.base_model_name)
architecture = config.architectures[0]
if "Llama" in architecture:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "</s>"
    DEFAULT_UNK_TOKEN = "</s>"
    print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )


key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = _get_submodules(model.base_model.model, key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)
        print(f"Replaced: {key}")

model = model.base_model.model

output_name = (
    script_args.adapter_model_name if script_args.adapter_model_name is not None else script_args.checkpoint_dir
)
if output_name.endswith("/"):
    output_name = output_name[:-1]
output_name += "_merged"
print(f"Output name: {output_name}")
# input("Ok?")

model.save_pretrained(f"{output_name}")
tokenizer.save_pretrained(f"{output_name}")

print(f"Saved model to {script_args.output_name}")
