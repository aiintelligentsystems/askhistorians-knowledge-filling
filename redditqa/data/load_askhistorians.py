from os import path

import datasets as ds
import pandas as pd

from redditqa.config import DATASETS_CACHE_DIR_PATH
from redditqa.data import util
from redditqa.data.dataset_split import split_dataset

ASK_HISTORIANS_FILENAME = "AskHistorians_question_filter_probabilities"


def load_askhistorians(yes_prob_quantile=0.5):
    # Load the dataset (with smart filter probabilities)
    askhistorians = ds.load_from_disk(path.join(DATASETS_CACHE_DIR_PATH, ASK_HISTORIANS_FILENAME))

    # Filter by smart filter probabilities
    askhistorians = askhistorians.map(_parse_graded_output)
    yes_quantile = pd.Series(askhistorians["prob_good"]).quantile(yes_prob_quantile)
    dataset = askhistorians.filter(lambda row: row["prob_good"] > yes_quantile)

    # Train test split
    dataset = dataset.map(util.create_deterministic_id)
    dataset = split_dataset(dataset, "splits/askhistorians_split.json")

    return dataset


def _parse_graded_output(row):
    graded_output = {entry["token_str"]: entry["probability"] for entry in row["graded_output"]}
    return {
        "prob_good": max(graded_output["Yes"], graded_output["yes"], graded_output["y"]),
        "prob_bad": max(graded_output["No"], graded_output["no"], graded_output["n"]),
    }
