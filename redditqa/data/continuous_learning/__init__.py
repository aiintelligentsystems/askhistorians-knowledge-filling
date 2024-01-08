import datasets as ds

from redditqa.data.continuous_learning import ultrachat, ultrafeedback


def add_continuous_learning_dataset(dataset, task, subset, tokenizer):
    if task == "sft":
        cl_dataset = ultrachat.prepare_dataset(tokenizer=tokenizer, subset=subset)
    elif task == "dpo":
        cl_dataset = ultrafeedback.prepare_dataset(tokenizer=tokenizer, subset=subset)

    dataset_merged = ds.DatasetDict()
    for split in dataset.keys():
        if split == "test":
            dataset_merged[split] = dataset[split]
        else:
            dataset_merged[split] = ds.concatenate_datasets([dataset[split], cl_dataset[split]]).shuffle(42)

    return dataset_merged
