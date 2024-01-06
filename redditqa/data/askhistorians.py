import datasets as ds
import pandas as pd

from redditqa.config import DATASETS_CACHE_DIR_PATH
from redditqa.data import pair_generation


def _return_prompt_and_responses(row) -> dict[str, str]:
    prompt_template = "<|ASKHIST|> Question: %question\nAnswer: "
    return {
        "prompt": prompt_template.replace("%question", row["question_title"]),
        "chosen": row["response_j"],
        "rejected": row["response_k"],
    }

def _parse_graded_output(row):
    graded_output = {
        entry["token_str"]: entry["probability"]
        for entry in row["graded_output"]
    }
    return {"prob_good": max(graded_output["Yes"], graded_output["yes"], graded_output["y"]), 
            "prob_bad": max(graded_output["No"], graded_output["no"], graded_output["n"])}


def load_dataset(
        filepath="AskHistorians_question_filter_probabilities", 
        yes_prob_quantile=0.75, 
        paired=False) -> ds.Dataset:
    """
    Loads RedditQA askHistorians dataset.
    Filters questions according to the smart filter probabilities.
    """

    askhistorians = ds.load_from_disk(DATASETS_CACHE_DIR_PATH + filepath)
    askhistorians = askhistorians.map(_parse_graded_output)
    yes_quantile = pd.Series(askhistorians["prob_good"]).quantile(yes_prob_quantile)
    askhistorians_filtered = askhistorians.filter(lambda row: row["prob_good"] > yes_quantile)

    if paired:
        # The dataset is converted to a dictionary with the following structure:
        # {
        #     'prompt': List[str],
        #     'chosen': List[str],
        #     'rejected': List[str],
        # }

        # Prompts are structured as follows:
        #   "<|ASKHIST|> Question: %question\nAnswer: "
        # TODO: replace with proper train/val/test split
        train_valid = askhistorians_filtered.train_test_split(test_size=0.1)["train"]
        train_valid = train_valid.train_test_split(test_size=0.1)

        train_data = train_valid["train"]
        valid_data = train_valid["test"]

        train_data = pair_generation.apply(train_data)
        valid_data = pair_generation.apply(valid_data)

        dataset = ds.DatasetDict(
            {
                "train": train_data,
                "eval": valid_data,
            }
        )
        dataset = dataset.map(_return_prompt_and_responses)
        return dataset

    return askhistorians_filtered
