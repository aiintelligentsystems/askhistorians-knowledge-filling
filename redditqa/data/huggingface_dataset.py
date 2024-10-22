import os

import datasets as ds
import pandas as pd

from redditqa.config import DATASETS_CACHE_DIR_PATH
from redditqa.data.util import create_deterministic_id, mask_links, replace_html_symbols


def load_reddit_dataset(dataset_file: str):
    datasets = ds.Dataset.from_pandas(pd.read_json(dataset_file, lines=True, orient="records"))

    # Trick to enable caching: Save and load the dataset to make sure it has a cache file
    # This means that the dataset is only loaded once and then loaded from the cache
    # When the csv file changes, one must manually delete the cache file or turn off caching manually
    cache_dir = os.path.join(DATASETS_CACHE_DIR_PATH, os.path.basename(dataset_file))
    datasets.save_to_disk(cache_dir)
    datasets = ds.load_from_disk(cache_dir)

    # Create ids
    datasets = datasets.map(create_deterministic_id)

    # Basic Preprocessing
    datasets = datasets.map(mask_links)
    datasets = datasets.map(replace_html_symbols)

    return datasets
