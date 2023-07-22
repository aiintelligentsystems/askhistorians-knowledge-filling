from typing import List

import evaluate

from redditqa.utils import prefix_dict


def calculate(predictions: List[str], references: List[str]):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {
        **prefix_dict(bleu_score, "bleu"),
        **prefix_dict(rouge_score, "rouge"),
        **prefix_dict(bertscore_score, "bertscore"),
    }
