import argparse
import json

from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from redditqa.dataset import load_reddit_dataset, preprocess_best_answer
from redditqa.evaluation import metrics

# Argparse
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_path",
    type=str,
    default="/scratch1/jhoff/llama-7b",
)
argparser.add_argument("--split", type=str, default="eval")
argparser.add_argument("--subset", type=int, default=1000)
argparser.add_argument("--output", type=str, default="/scratch1/jhoff/eval_output_baseline.json")
args = argparser.parse_args()

# Load the dataset
dataset = load_reddit_dataset(args.split)
dataset = dataset.map(preprocess_best_answer, batched=False)

# Load the generation model
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generation args
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "min_length": 32,
    "max_length": 128,
}


if args.subset:
    dataset = dataset.select(range(args.subset))

# Generate answers for the dataset
generations = []
for row in tqdm(dataset):
    generation = pipeline(row["prompt"], **generation_kwargs)
    generation = generation[0]["generated_text"]
    generation = generation.replace(row["prompt"], "")
    generations.append(generation)

# Evaluate the generations
metric_results = metrics.calculate(references=dataset["answer"], predictions=generations)

data = {
    "model": args.model_path,
    "split": args.split,
    "metrics": metric_results,
    "questions": dataset["prompt"],
    "generations": generations,
    "references": dataset["answer"],
}
print(data)
with open(args.output, "w") as f:
    json.dump(data, f)
