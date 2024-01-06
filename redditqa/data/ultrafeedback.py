import datasets as ds
import re

from redditqa.config import DATASETS_CACHE_DIR_PATH


def _prepare_sample(example, tokenizer, assistant_prefix="<|assistant|>\n"):
    """Prepare the text from a sample of the ultrafeedback dataset."""
    # taken from
    # https://github.com/huggingface/alignment-handbook/blob/e316174e1c6188ed45f9effa7a6e7d0081bf51d4/src/alignment/data.py#L59C1-L77C1
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("chosen", "rejected")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["chosen"][0])
        # TODO: handle case where chosen/rejected also have system messages
        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

    return example

def prepare_dataset(
        tokenizer,
        filepath="ultrafeedback_binarized",
        subset=None
        ) -> ds.Dataset:
    """
    Prepares ultrachat 200k dataset.
    """
    ultrafeedback = ds.load_from_disk(DATASETS_CACHE_DIR_PATH + filepath)

    dataset = ds.DatasetDict()
    rename_splits = {
        "train_prefs":"train",
        "test_prefs":"eval"
    }

    for split in ['train_prefs', 'test_prefs']:
        ultrafeedback_split = ultrafeedback[split].select(range(subset))
        dataset[rename_splits[split]] = ultrafeedback_split.map(
            _prepare_sample,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=ultrafeedback_split.column_names
        )
        dataset[rename_splits[split]] = dataset[rename_splits[split]].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    return dataset
