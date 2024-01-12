from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
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


class Preference(Enum):
    PREFER_MODEL = 1
    PARITY = 2
    PREFER_BASELINE = 3


@dataclass
class PreferenceOverview:
    counts: Dict[Preference, int]
    freqs: Dict[Preference, float]

    def __str__(self):
        return f"""
Prefer Model: {self.freqs[Preference.PREFER_MODEL] * 100:.2f}%
Prefer Baseline: {self.freqs[Preference.PREFER_BASELINE] * 100:.2f}%
Parity: {self.freqs[Preference.PARITY] * 100:.2f}%
        """.strip()


def gpt4_compare(question: List[str], answer: List[str], answer_baseline: List[str]):
    preferences = []
    for q, a, a_baseline in zip(question, answer, answer_baseline):
        preferences.append(_gpt4_preference(q, a, a_baseline))

    counts = Counter(preferences)
    counts = {val: counts[val] for val in Preference}
    freqs = {k: v / len(preferences) for k, v in counts.items()}

    return PreferenceOverview(counts=counts, freqs=freqs)


def _gpt4_preference(question: str, answer: str, answer_baseline: str) -> Preference:
    switch_order = np.random.choice([True, False])

    user_prompt = PROMPT_TEMPLATE.replace("%QUESTION", question)
    if not switch_order:
        user_prompt = user_prompt.replace("%ANSWER1", answer)
        user_prompt = user_prompt.replace("%ANSWER2", answer_baseline)
    else:
        user_prompt = user_prompt.replace("%ANSWER1", answer_baseline)
        user_prompt = user_prompt.replace("%ANSWER2", answer)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].message.content

    # Convert to preference
    if "[[A]]" in answer and "[[B]]" in answer:
        preference = Preference.PARITY
    elif "[[A]]" in answer:
        preference = Preference.PREFER_MODEL if not switch_order else Preference.PREFER_BASELINE
    elif "[[B]]" in answer:
        preference = Preference.PREFER_BASELINE if not switch_order else Preference.PREFER_MODEL
    else:
        preference = Preference.PARITY

    print("GPT Answer:", answer)
    print("GPT Preference:", preference)

    return preference
