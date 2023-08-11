import re

import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from redditqa.dataset.preprocessing.links import mask_links

DATASETS_BASE_PATH = "/scratch1/jhoff"


def binary_comparison(answers):
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


def preprocess_pair_generation(examples):
    """Returns paired answers (j is better than k). Note that this returns more examples (one for each pair per question)."""

    MAX_PAIRS_PER_QUESTION = 10
    n_samples = len(examples["answer_link_id"])

    # Initialize empty lists for new samples
    new_examples = {"question_title": [], "response_j": [], "response_k": [], "score_j": [], "score_k": []}

    # Overwrite all other keys
    for key in examples:
        new_examples[key] = []

    for sample_id in range(n_samples):
        # Get pairs where first is always the better one
        pairs = binary_comparison(examples["answers"][sample_id])

        # Sample if we get more pairs than maximum
        if len(pairs) > MAX_PAIRS_PER_QUESTION:
            indices = np.random.choice(list(range(len(pairs))), MAX_PAIRS_PER_QUESTION, replace=False)
            pairs = [pairs[i] for i in indices]

        # Construct the samples
        for pair in pairs:
            for key in examples:
                new_examples[key].append(examples[key][sample_id])
            new_examples["response_j"].append(pair[0]["body"])
            new_examples["response_k"].append(pair[1]["body"])
            new_examples["score_j"].append(pair[0]["score"])
            new_examples["score_k"].append(pair[1]["score"])
    return new_examples


def preprocess_best_answer(example):
    submission_title = example["question_title"]
    comments = example["answers"]
    comments = sorted(comments, key=lambda k: k["answer_score"])
    answer = comments[-1]["answer_body"]

    prompt = f"Question: {submission_title}\nAnswer: "
    full_text = f"Question: {submission_title}\nAnswer: {answer}"

    return {"full_text": full_text, "prompt": prompt, "question": submission_title, "answer": answer}


def load_reddit_dataset(split=None, pairs=False):
    if split is None:
        datasets = ds.DatasetDict(
            {
                "train": ds.Dataset.from_pandas(
                    pd.read_json(f"{DATASETS_BASE_PATH}/elif_preproc_train.jsonl", lines=True)
                ),
                "eval": ds.Dataset.from_pandas(
                    pd.read_json(f"{DATASETS_BASE_PATH}/elif_preproc_eval.jsonl", lines=True)
                ),
                "test": ds.Dataset.from_pandas(
                    pd.read_json(f"{DATASETS_BASE_PATH}/elif_preproc_test.jsonl", lines=True)
                ),
            }
        )
    else:
        datasets = ds.DatasetDict(
            {
                split: ds.Dataset.from_pandas(
                    pd.read_json(f"{DATASETS_BASE_PATH}/elif_preproc_{split}.jsonl", lines=True)
                ),
            }
        )

    # Trick to enable caching: Save and load the dataset to make sure it has a cache file
    datasets.save_to_disk(f"{DATASETS_BASE_PATH}/reddit_dataset_cached")
    datasets = ds.load_from_disk(f"{DATASETS_BASE_PATH}/reddit_dataset_cached")

    # Mask links
    datasets = datasets.map(mask_links)

    if pairs:
        datasets = datasets.map(preprocess_pair_generation, batch_size=10, batched=True)
        datasets = datasets.remove_columns(["answers"])
    else:
        pass

    datasets = datasets.shuffle(seed=42)

    if split is not None:
        return datasets[split]
    else:
        return datasets


if __name__ == "__main__":
    # print(len(load_reddit_dataset("eval")))
    data = load_reddit_dataset("test")
    print(data)
    # print(next(iter(load_reddit_dataset("test"))))
