import datasets as ds


def apply(dataset: ds.Dataset | ds.DatasetDict, remove_columns=True):
    dataset = dataset.map(generate_best_answer, batched=False)

    if remove_columns:
        dataset = dataset.remove_columns(["answers"])

    return dataset


def generate_best_answer(example):
    submission_title = example["question_title"]
    comments = example["answers"]
    comments = sorted(comments, key=lambda k: k["answer_score"])
    answer = comments[-1]["answer_body"]

    prompt = f"Question: {submission_title}\nAnswer: "
    full_text = f"Question: {submission_title}\nAnswer: {answer}"

    return {
        "full_text": full_text,
        "prompt": prompt,
        "question": submission_title,
        "answer": answer,
    }
