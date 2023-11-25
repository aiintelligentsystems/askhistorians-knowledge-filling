

DATASETS_BASE_PATH = "/scratch1/redditqa/data"


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
