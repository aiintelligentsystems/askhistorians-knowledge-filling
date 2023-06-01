import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datasets as ds

DATASETS_BASE_PATH = "/scratch1/jhoff/"

def binary_comparison(answers):
    """Returns tuples of answers, first always best"""
    pairs = []
    
    for i in range(len(answers)-1):
        for j in range(i+1, len(answers)):
            if answers[i]["score"]>answers[j]["score"]:
                pairs.append((answers[i]["body"], answers[j]["body"]))
            elif answers[i]["score"]<answers[j]["score"]:
                pairs.append((answers[j]["body"], answers[i]["body"]))
    return pairs

def preprocess(examples):
    """Returns paired answers (j is better than k). Note that this returns more examples (one for each pair per question)."""
    
    MAX_PAIRS_PER_QUESTION = 10
    n_samples = len(examples["link_id"])
    
    # initialize empty lists for new samples
    new_examples = {"submission_title": [], "response_j": [], "response_k": []}
    for key in examples:
        new_examples[key] = []
    
    for sample_id in range(n_samples):
        # get pairs where first is always the better one
        pairs = binary_comparison(examples["comments"][sample_id])
        
        # sample if we get more pairs than maximum
        if len(pairs) > MAX_PAIRS_PER_QUESTION:
            indices = np.random.choice(list(range(len(pairs))), MAX_PAIRS_PER_QUESTION, replace=False)
            pairs = [pairs[i] for i in indices]
        
        # construct the samples
        for pair in pairs:
            for key in examples:
                new_examples[key].append(examples[key][sample_id])
            new_examples["response_j"].append(pair[0])
            new_examples["response_k"].append(pair[1])
    return new_examples

def get_datasets():
    datasets = ds.DatasetDict(
        {
            "train": ds.Dataset.from_pandas(pd.read_json(f"{DATASETS_BASE_PATH}/elif_train.jsonl", lines=True)),
            "validation": ds.Dataset.from_pandas(pd.read_json(f"{DATASETS_BASE_PATH}/scratch1/jhoff/elif_val.jsonl", lines=True)),
            "test": ds.Dataset.from_pandas(pd.read_json(f"{DATASETS_BASE_PATH}/scratch1/jhoff/elif_test.jsonl", lines=True)),
        }
    )
    datasets = datasets.map(preprocess, batch_size=10, batched=True)
    dataset = datasets.remove_columns(["comments"])
    dataset = dataset.shuffle(seed=42)

    return datasets