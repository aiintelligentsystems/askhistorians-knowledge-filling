import datasets as ds
import pandas as pd

from redditqa.config import DATASETS_CACHE_DIR_PATH
from redditqa.data import pair_generation


def _parse_graded_output(row):
    graded_output = {
        entry["token_str"]: entry["probability"]
        for entry in row["graded_output"]
    }
    return {"prob_good": max(graded_output["Yes"], graded_output["yes"], graded_output["y"]), 
            "prob_bad": max(graded_output["No"], graded_output["no"], graded_output["n"])}


def _prepare_dpo_sample(row) -> dict[str, str]:
    prompt_template = "<|ASKHIST|> Question: %question\nAnswer: "
    return {
        "prompt": prompt_template.replace("%question", row["question_title"]),
        "chosen": row["response_j"],
        "rejected": row["response_k"],
    }


def _prepare_sft_sample(row):
    """Prepare the text from a sample of the redditqa dataset."""
    submission_title = row["question_title"]
    comments = row["answers"]
    comments = sorted(comments, key=lambda k: k["answer_score"])
    answer = comments[-1]["answer_body"]
    text = f"Question: {submission_title}\nAnswer: {answer}"
    return dict(text=text)

def load_dataset(
        filepath="AskHistorians_question_filter_probabilities", 
        yes_prob_quantile=0.75,
        split=True,
        task=None,
        ) -> ds.Dataset | ds.DatasetDict:
    """
    Loads RedditQA askHistorians dataset.
    Filters questions according to the smart filter probabilities.

    `task` is in `[None, 'sft', 'dpo']`.
    """

    askhistorians = ds.load_from_disk(DATASETS_CACHE_DIR_PATH + filepath)
    askhistorians = askhistorians.map(_parse_graded_output)
    yes_quantile = pd.Series(askhistorians["prob_good"]).quantile(yes_prob_quantile)
    dataset = askhistorians.filter(lambda row: row["prob_good"] > yes_quantile)

    if split:
        # TODO: replace with proper train/val/test split
        train_valid = dataset.train_test_split(test_size=0.1)["train"]
        train_valid = train_valid.train_test_split(test_size=0.1)

        dataset = ds.DatasetDict(
            {
                "train": train_valid["train"],
                "eval": train_valid["test"],
            }
        )

    if task == 'sft':
        dataset = dataset.map(
            _prepare_sft_sample,
            remove_columns=dataset.column_names
            )
        return dataset
        
    elif task == 'dpo':
        # The dataset is converted to a dictionary with the following structure:
        # {
        #     'prompt': List[str],
        #     'chosen': List[str],
        #     'rejected': List[str],
        # }

        # Prompts are structured as follows:
        #   "<|ASKHIST|> Question: %question\nAnswer: "

        dataset = pair_generation.apply(dataset)
        dataset = dataset.map(_prepare_dpo_sample)
        return dataset
    
    else:
        return dataset
