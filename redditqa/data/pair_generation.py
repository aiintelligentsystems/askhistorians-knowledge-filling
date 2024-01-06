import re
from typing import List

import datasets as ds
import numpy as np

MAX_PAIRS_PER_QUESTION = 10


def apply(dataset: ds.Dataset | ds.DatasetDict, remove_columns=True, score_margin=None):
    dataset = dataset.map(preprocess_pair_generation, fn_kwargs={"score_margin": score_margin}, batched=True)

    if remove_columns:
        dataset = dataset.remove_columns(["answers"])

    return dataset


def preprocess_pair_generation(examples, score_margin=None):
    """Returns paired answers (j is better than k). Note that this returns more examples (one for each pair per question)."""

    n_samples = len(examples["question_title"])

    # Initialize empty lists for new samples
    new_examples = {
        "question_title": [],
        "response_j": [],
        "response_k": [],
        "score_j": [],
        "score_k": [],
    }

    # Overwrite all other keys
    for key in examples:
        new_examples[key] = []

    for sample_idx in range(n_samples):
        # Get pairs where first is always the better one
        pairs = _binary_comparison(examples["answers"][sample_idx])

        if score_margin is not None:
            pairs = [pair for pair in pairs if abs(pair[0]["score"] - pair[1]["score"]) >= score_margin]

        # Sample if we get more pairs than maximum
        if len(pairs) > MAX_PAIRS_PER_QUESTION:
            indices = np.random.choice(list(range(len(pairs))), MAX_PAIRS_PER_QUESTION, replace=False)
            pairs = [pairs[i] for i in indices]

        # Construct the samples
        for pair in pairs:
            for key in examples:
                new_examples[key].append(examples[key][sample_idx])
            new_examples["response_j"].append(pair[0]["body"])
            new_examples["response_k"].append(pair[1]["body"])
            new_examples["score_j"].append(pair[0]["score"])
            new_examples["score_k"].append(pair[1]["score"])

    return new_examples


def _binary_comparison(answers):
    """Returns tuples of answers, first always best"""
    pairs = []

    for i in range(len(answers) - 1):
        for j in range(i + 1, len(answers)):
            # Get scores and data
            answer_i = answers[i]
            answer_j = answers[j]
            answer_score_i = answer_i["answer_score"]
            answer_score_j = answer_j["answer_score"]
            answer_body_i = answers[i]["answer_body"]
            answer_body_j = answers[j]["answer_body"]
            data_i = {"score": answer_score_i, "body": answer_body_i}
            data_j = {"score": answer_score_j, "body": answer_body_j}

            # Create pair
            if answer_score_i > answer_score_j:
                pairs.append((data_i, data_j))
            elif answer_score_i < answer_score_j:
                pairs.append((data_j, data_i))
    return pairs
