import datasets as ds
import pandas as pd

from redditqa.config import DATASETS_CACHE_DIR_PATH

def _prepare_sample(example, tokenizer):
    """Prepare the text from a sample of the ultrachat dataset."""
    # taken from
    # https://github.com/huggingface/alignment-handbook/blob/e316174e1c6188ed45f9effa7a6e7d0081bf51d4/src/alignment/data.py#L35C1-L42C10
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        # weird system prompt issue: https://github.com/huggingface/alignment-handbook/issues/52
        messages.insert(0, {"role": "system", "content": ""})
    return dict(text=tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    ))

def prepare_dataset(
        tokenizer,
        filepath="ultrachat_200k",
        subset=None
        ) -> ds.Dataset:
    """
    Prepares ultrachat 200k dataset.
    """
    ultrachat = ds.load_from_disk(DATASETS_CACHE_DIR_PATH + filepath)

    dataset = ds.DatasetDict()
    rename_splits = {
        "train_sft":"train",
        "test_sft":"eval"
    }

    for split in ['train_sft', 'test_sft']:
        ultrachat_split = ultrachat[split].select(range(subset))
        dataset[rename_splits[split]] = ultrachat_split.map(
            _prepare_sample,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=ultrachat_split.column_names
        )
    return dataset