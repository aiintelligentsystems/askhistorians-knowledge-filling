import datasets as ds
import pandas as pd

from redditqa import config
from redditqa.config import DATASETS_CACHE_DIR_PATH

def load_dataset(filepath="AskHistorians_question_filter_probabilities", yes_prob_quantile=0.75):
    """
    Loads RedditQA askHistorians dataset.
    Filters questions according to the smart filter probabilities.
    """
    config.DATASETS_CACHE_DIR_PATH
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