from redditqa.data.dataset_split import split_dataset
from redditqa.data.huggingface_dataset import load_reddit_dataset
from redditqa.data.util import (
    fix_eli5_question_title,
    get_answer_len_filter,
    remove_answers_with_edit_marker,
)


def load_eli5():
    dataset = load_reddit_dataset("/scratch1/redditqa/data/eli5/eli5.jsonl")

    # Apply dataset split
    dataset = split_dataset(dataset, "splits/eli5_split.json")

    # Fix question title
    dataset = dataset.map(fix_eli5_question_title, batched=False)
    # Remove answers with edit marker so that we don't learn to output "edit: ..."
    dataset = dataset.map(remove_answers_with_edit_marker)

    # Stat filtering
    dataset = dataset.filter(lambda row: len(row["question_title"]) > 50)
    dataset = dataset.filter(lambda row: row["question_score"] > 20)
    dataset = dataset.map(
        get_answer_len_filter(min_len=20),
    )
    dataset = dataset.filter(lambda row: len(row["answers"]) > 3)

    return dataset
