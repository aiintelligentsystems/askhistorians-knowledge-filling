import json

import datasets as ds


def split_dataset(dataset: ds.Dataset, split_definition_file: str) -> ds.DatasetDict:
    """
    Split a dataset based on the questions per split as defined in the split definition file
    """

    # Load the split definition
    with open(split_definition_file) as f:
        ids_per_split = json.load(f)

    # Create the dataset dict
    dataset_dict = ds.DatasetDict(
        {
            "train": dataset.filter(_create_question_selector(ids_per_split["train"])),
            "eval": dataset.filter(_create_question_selector(ids_per_split["eval"])),
            "test": dataset.filter(_create_question_selector(ids_per_split["test"])),
        }
    )

    return dataset_dict


def _create_question_selector(ids):
    ids = set(ids)

    def select_questions(row):
        return row["id"] in ids

    return select_questions
