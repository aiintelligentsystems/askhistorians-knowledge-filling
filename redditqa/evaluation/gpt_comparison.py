import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import datasets as ds
import huggingface_hub
import torch
from accelerate import Accelerator
from datasets import load_dataset
from openai import OpenAI

client = OpenAI()


PROMPT_TEMPLATE = """
You are a seasoned historian tasked with evaluating responses to historical questions. 
Consider the following question and assess which of the two provided 
answers presents the most accurate and comprehensive information. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User Question]
Question: %QUESTION

[The Start of Assistant A's Answer]
%ANSWER1
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
%ANSWER2
[The End of Assistant B's Answer]
""".strip()


def compare(answers_left: List[str], answers_right: List[str], question: str):
    pass


def _gpt4_preference(ds_item):
    model1, model2 = get_model_order([model_a, model_b])
    user_prompt = PROMPT_TEMPLATE.replace("%QUESTION", ds_item["question_title"])
    user_prompt = user_prompt.replace("%ANSWER1", ds_item[model1])
    user_prompt = user_prompt.replace("%ANSWER2", ds_item[model2])

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].message.content

    ds_item["model-order"] = f"Answer1:{model1};Answer2:{model2}"
    ds_item["raw-gpt4-answer"] = answer

    # Convert to preference
    if "[[A]]" in answer and "[[B]]" in answer:
        ds_item["gpt4-preference"] = ""
    elif "[[A]]" in answer:
        ds_item["gpt4-preference"] = model1
    elif "[[B]]" in answer:
        ds_item["gpt4-preference"] = model2
    else:
        ds_item["gpt4-preference"] = ""

    return ds_item
