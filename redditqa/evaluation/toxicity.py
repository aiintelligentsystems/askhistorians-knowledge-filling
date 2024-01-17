from dataclasses import dataclass
from typing import List

import numpy as np
from transformers import pipeline

from redditqa.evaluation.util import Result, result_from_values


def calc(answers: List[str]) -> Result:
    toxicity_pipe = pipeline("text-classification", model="tomh/toxigen_roberta")
    results = [_run_toxicity_pipe(toxicity_pipe, answer) for answer in answers]
    return result_from_values(results)


def _run_toxicity_pipe(toxicity_pipe, text):
    try:
        result = toxicity_pipe(text, top_k=None)
        result = [r for r in result if r["label"] == "LABEL_1"][0]
    except:
        return 0.5

    return result["score"]
