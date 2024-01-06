import datasets as ds

from redditqa.data.continuous_learning import ultrachat
from redditqa.data.continuous_learning import ultrafeedback


def add_continuous_learning_dataset(dataset, task, subset, tokenizer):
    if task == "sft":
        cl_dataset = ultrachat.prepare_dataset(tokenizer=tokenizer, subset=subset)
    elif task == "dpo":
        cl_dataset = ultrafeedback.prepare_dataset(tokenizer=tokenizer, subset=subset)

    dataset = _concat_splits(dataset, cl_dataset)
    return dataset


def _concat_splits(dataset1: ds.DatasetDict, dataset2: ds.DatasetDict) -> ds.DatasetDict:
    """Concatenate two dataset dicts with the same splits."""
    dataset = ds.DatasetDict()
    for split in dataset1.keys():
        dataset[split] = ds.concatenate_datasets([dataset1[split], dataset2[split]])
    return dataset
