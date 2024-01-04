import os

import datasets as ds
import pandas as pd

from redditqa.data.util import mask_links

DATASETS_CACHE_DIR_PATH = "/scratch1/redditqa/cached_datasets"


def load_reddit_dataset(dataset_file: str):
    datasets = ds.Dataset.from_pandas(pd.read_json(dataset_file, lines=True, orient="records"))

    # Trick to enable caching: Save and load the dataset to make sure it has a cache file
    # This means that the dataset is only loaded once and then loaded from the cache
    # When the csv file changes, one must manually delete the cache file or turn off caching manually
    cache_dir = os.path.join(DATASETS_CACHE_DIR_PATH, os.path.basename(dataset_file))
    datasets.save_to_disk(cache_dir)
    datasets = ds.load_from_disk(cache_dir)

    # Preprocessing
    datasets = datasets.map(mask_links)

    return datasets


def load_redditqa_dataset(filepath="AskHistorians_question_filter_probabilities", yes_prob_quantile=0.75):
    """
    Loads RedditQA askHistorians dataset.
    Filters questions according to the smart filter probabilities.
    """
    def parse_graded_output(row):
        graded_output = {
            entry["token_str"]: entry["probability"]
            for entry in row["graded_output"]
        }
        return {"prob_good": max(graded_output["Yes"], graded_output["yes"], graded_output["y"]), 
                "prob_bad": max(graded_output["No"], graded_output["no"], graded_output["n"])}

    askhistorians = ds.load_from_disk(DATASETS_CACHE_DIR_PATH + filepath)
    askhistorians = askhistorians.map(parse_graded_output)
    yes_quantile = pd.Series(askhistorians["prob_good"]).quantile(yes_prob_quantile)
    askhistorians_filtered = askhistorians.filter(lambda row: row["prob_good"] > yes_quantile)
    return askhistorians_filtered