from typing import Literal

import datasets as ds

from redditqa.data import pair_generation
from redditqa.data.load_askhistorians import load_askhistorians
from redditqa.data.load_eli5 import load_eli5


def _prepare_dpo_sample(row) -> dict[str, str]:
    prompt_template = "<|REDDITQA|> Question: %question\nAnswer: "
    return {
        "prompt": prompt_template.replace("%question", row["question_title"]),
        "chosen": row["response_j"],
        "rejected": row["response_k"],
        "score_choosen": row["score_j"],
        "score_rejected": row["score_k"],
    }


def _prepare_sft_sample(row):
    """Prepare the text from a sample of the redditqa dataset."""
    submission_title = row["question_title"]
    comments = row["answers"]
    comments = sorted(comments, key=lambda k: k["answer_score"])
    answer = comments[-1]["answer_body"]
    text = f"<|REDDITQA|> Question: {submission_title}\nAnswer: {answer}"
    return dict(text=text)


def load_dataset(
    name: Literal["askhistorians", "eli5"],
    task=None,
    eval_subsample=None,
    score_margin=None,
) -> ds.Dataset | ds.DatasetDict:
    """
    Loads RedditQA askHistorians dataset.
    Filters questions according to the smart filter probabilities.

    `task` is in `[None, 'sft', 'dpo']`.

    For task="dpo", the dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
            'score_choosen': List[float],
            'score_rejected': List[float],
        }
    Prompts are structured as follows:
        "<|ASKHIST|> Question: %question\nAnswer: "
    """

    # Load the dataset (already split and preprocessed)
    if name == "askhistorians":
        dataset = load_askhistorians()
    elif name == "eli5":
        dataset = load_eli5()

    # Apply task-specific preprocessing
    if task == "sft":
        dataset = dataset.map(_prepare_sft_sample, remove_columns=dataset["train"].column_names)
    elif task == "dpo":
        dataset = pair_generation.apply(dataset, score_margin=score_margin)
        dataset = dataset.map(_prepare_dpo_sample, remove_columns=dataset["train"].column_names)

    # Subsample the evaluation set
    if eval_subsample is not None:
        dataset["eval"] = dataset["eval"].shuffle().select(range(eval_subsample))

    return dataset
