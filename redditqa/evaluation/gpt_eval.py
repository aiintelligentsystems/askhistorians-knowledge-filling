# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import random
from collections import Counter, defaultdict

# %%
from dataclasses import dataclass, field
from typing import Optional

import huggingface_hub
import openai
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from reddit_dataset import load_reddit_dataset
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    HfArgumentParser,
    TextGenerationPipeline,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

random.seed(42)

# %%
num_test_samples = 1000

model_path_baseline = "EleutherAI/pythia-6.9B-deduped"
model_path_trained = "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-20000_merged"

generation_kwargs_config = generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    # "eos_token_id": 100_000,
    "min_length": 32,
    "max_length": 128,
}

# %%
test_set = load_reddit_dataset("test")
test_set = test_set.select(range(num_test_samples))
test_questions = [f"Question: {x['submission_title']}\nAnswer: " for x in test_set]

test_questions[:5]

# %%
results = {
    model_path_baseline: {},
    model_path_trained: {},
}

# %%
for model_path in [model_path_trained]:
    model = GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.config.pad_token_id = model.config.eos_token_id
    model = model.cuda().eval()

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda:0")

    generation_kwargs = {
        **generation_kwargs_config,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for question in tqdm(test_questions):
        result = pipeline(question, **generation_kwargs)
        result = result[0]["generated_text"]

        results[model_path][question] = result

# %%
results

# %%
results2 = json.load(open("results-eval-1.json", "r"))
results_all = {**results, **results2}

# %%
results_new_format = defaultdict(dict)

for model, rows in results_all.items():
    for question, result in rows.items():
        results_new_format[question][model] = result.replace(question, "").strip()

# %%
results_new_format = {q: answers for q, answers in results_new_format.items() if len(answers) == 2}

# %%
len(results_new_format), results_new_format

# %%
json.dump(results_new_format, open("data.json", "w"))

# %%
results_new_format

# %%


# %%
openai.api_key = input("Enter your OpenAI API key: ")

# %%
baseline_name = "EleutherAI/pythia-6.9B-deduped"
trained_name = "/scratch1/jhoff/checkpoints/finetuned-pythia-6.9B-deduped/checkpoint-20000_merged"

data = json.load(open("data.json"))

# %%
PROMPT_TEMPLATE = """You are an expert reddit user. 

For the following question, answer which of the two answers is the best answer.

Question: %QUESTION

Answer 1: %ANSWER1

Answer 2: %ANSWER2

The best answer is (1 or 2):"""


def compare_answers(question, answer1, answer2):
    prompt = PROMPT_TEMPLATE.replace("%QUESTION", question).replace("%ANSWER1", answer1).replace("%ANSWER2", answer2)

    # Prompt model
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )

    # Get the answer
    answer = response.choices[0].text
    print(f"{question}")
    print(f"Model 1: {answer1} <<<<<")
    print(f"Model 2: {answer2} <<<<<")
    print(f"Raw GPT Answer: {answer}")
    print("-" * 20)
    print("-" * 20)

    # Convert to number
    if "1" in answer and "2" in answer:
        return None
    elif "1" in answer:
        return 1
    elif "2" in answer:
        return 2
    else:
        return None


def compare_answers_random(question, answer1, answer2):
    if random.uniform(0, 1) < 0.5:
        answer = compare_answers(question, answer1, answer2)
        if answer != None:
            return answer
    else:
        answer = compare_answers(question, answer2, answer1)
        if answer != None:
            return 3 - answer
    return None


# %%
for question, values in data.items():
    if "preference" in values:
        print(f"{values['preference']}")

# %%
num_answers = len(data)

for question, answers in tqdm(list(data.items())[:num_answers]):
    preference = compare_answers_random(question, answers[baseline_name], answers[trained_name])
    if preference != None:
        data[question]["preference"] = [baseline_name, trained_name][preference - 1]

# %%
Counter([values.get("preference", None) for values in data.values()])
