import json

import datasets as ds

QUESTIONS_SPLIT_DEFINITION_FILE = "train_test_dev_questions.json"


def split_dataset(dataset: ds.Dataset) -> ds.DatasetDict:
    """
    Split a dataset based on the questions per split as defined in the split definition file
    """

    # Load the split definition
    with open(QUESTIONS_SPLIT_DEFINITION_FILE) as f:
        titles_per_split = json.load(f)

    # Create the dataset dict
    dataset_dict = ds.DatasetDict(
        {
            "train": dataset.filter(_create_question_selector(titles_per_split["train"]), batched=False),
            "eval": dataset.filter(_create_question_selector(titles_per_split["eval"]), batched=False),
            "test": dataset.filter(_create_question_selector(titles_per_split["test"]), batched=False),
        }
    )

    return dataset_dict


def _create_question_selector(question_titles):
    question_titles = set(question_titles)

    def select_questions(row):
        return row["question_title"] in question_titles

    return select_questions
