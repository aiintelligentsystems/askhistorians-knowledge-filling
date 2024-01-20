from dataclasses import dataclass
from typing import List
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline
from peft import PeftModel

@dataclass
class Result:
    mean: float
    std: float

    def __str__(self):
        return f"{self.mean:.2f} ± {self.std:.2f}"


def result_from_values(values: List[float]) -> Result:
    return Result(mean=np.mean(values), std=np.std(values))


def load_model_and_tokenizer(model_name: str, adapter_name: str | None):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Load adapter if exists
    if adapter_name is not None:
        model_merged = PeftModel.from_pretrained(
            model,
            adapter_name,
        )
        model = model_merged.merge_and_unload()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
