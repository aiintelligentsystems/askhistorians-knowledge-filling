import argparse
import json
import logging
import os
import pickle as pkl
from itertools import product
from uuid import uuid4

from lm_eval import evaluator
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="eval_results")
args = parser.parse_args()

tasks = [
    "arc_easy",
    # "arc_challenge",
    # "boolq",
    # "hellaswag",
    # "openbookqa",
    # "winogrande",
    # "truthfulqa_gen",
    # "truthfulqa_mc",
    "triviaqa",
]

models = [
    # "gpt2",
    # "EleutherAI/pythia-160M-deduped",
    # "EleutherAI/pythia-410M-deduped",
    # "EleutherAI/pythia-1B-deduped",
    # "EleutherAI/pythia-1.4B-deduped",
    # "EleutherAI/pythia-2.8B-deduped",
    "EleutherAI/pythia-6.9B-deduped",
    # "EleutherAI/pythia-12B-deduped",
    # "/scratch1/jhoff/checkpoints/finetuned-pythia-2.8B-deduped/checkpoint_5000_merged",
    # "/scratch1/jhoff/checkpoints/finetuned-pythia-2.8B-deduped/checkpoint_10000_merged",
    # "/scratch1/jhoff/checkpoints/finetuned-pythia-2.8B-deduped/checkpoint_15000_merged",
    "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-5000_merged",
    "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-10000_merged",
    "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-15000_merged",
    "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-20000_merged",
    "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-25000_merged",
]

for model in tqdm(models):
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        # model_args=f"pretrained={model}",
        model_args=f"pretrained='{model}',load_in_8bit=True,device_map_option='auto',use_accelerate=True",
        tasks=tasks,
        batch_size=1,
        max_batch_size=1,
        device="cuda",
    )

    fname_base = f'{args.out_dir}/results-{model.replace("/", "-")}-{uuid4().hex}'
    with open(f"{fname_base}.pkl", "wb") as f:
        pkl.dump(results, f)
    with open(f"{fname_base}.json", "w") as f:
        json.dump(results, f, indent=2)
