import json
from dataclasses import dataclass, field
from os.path import basename, join
from typing import List, Optional
import numpy as np

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline
import datasets as ds
import wandb
import os
from evaluate import load

from redditqa.data.loader import load_dataset
from redditqa.evaluation import (
    human_comparison,
    llm_comparison,
    textstat_helper,
    toxicity,
)
from redditqa.evaluation.llm_comparison import PreferenceOverview
from redditqa.evaluation.perplexity import calculate_perplexity_for_expected_tokens
from redditqa.evaluation.util import Result, load_model_and_tokenizer


@dataclass
class EvalConfig:
    tasks: list[str]
    model_name: Optional[str] = field()
    adapter_name: Optional[str] = field()
    dataset_name: Optional[str] = field()
    output_dir: Optional[str] = field()
    wandb_project: Optional[str] = field()
    baseline_model_name: Optional[str] = field()
    split: Optional[str] = field(default="test")
    n_questions: Optional[int] = field(default=100)


@dataclass
class QuestionAnswerAnswer:
    question: str
    answer: str
    baseline_answer: str

    def __str__(self) -> str:
        return f"Question: {self.question}\nAnswer: {self.answer}\nBaseline Answer: {self.baseline_answer}"


@dataclass
class EvalReport:
    config: EvalConfig
    text_complexity: Result
    reading_time: Result
    toxicity: Result
    comparison_sheet_path: str
    baseline_text_complexity: Result
    baseline_reading_time: Result
    baseline_toxicity: Result
    gpt4_preference: PreferenceOverview
    data: List[QuestionAnswerAnswer]

    def as_multiline_string(self):
        data_str = "\n------\n".join(str(d) for d in self.data)
        return f"""\
Comparison between {basename(self.config.model_name)} and {basename(self.config.baseline_model_name)}

Model: {self.config.model_name}
Baseline Model: {self.config.baseline_model_name}
Dataset: {self.config.dataset_name}
Split: {self.config.split}
N questions: {self.config.n_questions}

Model: 
Text complexity: {self.text_complexity}
Reading time: {self.reading_time}
Toxicity: {self.toxicity}

Baseline: 
Text complexity: {self.baseline_text_complexity}
Reading time: {self.baseline_reading_time}
Toxicity: {self.baseline_toxicity}

GPT4 Preference:
{self.gpt4_preference}

Comparison sheet: {self.comparison_sheet_path}

Data:

{data_str}
        """.strip()

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.as_multiline_string())

def _generate_answers(questions: List[str], model_name: str, adapter_name: str | None, generation_args: dict) -> List[str]:
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_name)
    # Create the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate answers
    generation_args = {
        **generation_args,
        "pad_token_id": tokenizer.pad_token_id,
    }
    answers = pipe(questions, **generation_args)
    answers = [a[0]["generated_text"].replace(q, "").strip() for q, a in zip(questions, answers)]

    return answers


def main():
    parser = HfArgumentParser(EvalConfig)
    args = parser.parse_args_into_dataclasses()[0]
    args.adapter_name = None
    wandb.init(entity="reddit-qa", project=args.wandb_project, name=f"eval_{os.path.basename(os.path.dirname(args.model_name))}")
    print(f"Wandb run can be found here: {wandb.run.get_url()}")

    if any(task in args.tasks for task in ["base", "gpt4-preference"]):
        # Load the dataset
        dataset = load_dataset(name=args.dataset_name, task="sft").shuffle(42)
        dataset = dataset[args.split]
        dataset = dataset.select(range(args.n_questions))

        # Generate answers
        generation_args = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "min_length": 32,
            "max_length": 1024,
        }
        answers = _generate_answers(dataset["prompt"], args.model_name, args.adapter_name, generation_args)

        # Apply metrics
        if "base" in args.tasks:
            textstat_result = textstat_helper.calc(answers)        
            toxicity_result = toxicity.calc(answers)

        # If baseline model exists, compare with it
        if args.baseline_model_name is not None:
            baseline_answers = _generate_answers(dataset["prompt"], args.baseline_model_name, None)

            baseline_textstat_result = textstat_helper.calc(baseline_answers)
            baseline_toxicity_result = toxicity.calc(baseline_answers)

            # Create sheet for human comparison
            if "create-human-comp" in args.tasks:
                sheet_path = human_comparison.create_sheet(
                    output_path=args.output_dir,
                    name=f"Model Comparison {basename(args.model_name)} with {basename(args.baseline_model_name)}",
                    questions=dataset["question"],
                    answers=answers,
                    baseline_answers=baseline_answers,
                )
        else:
            baseline_answers = [""] * len(answers)
            baseline_textstat_result = None
            baseline_toxicity_result = None
            sheet_path = None

        # Calculate gpt4 preference
        if "gpt4-preference" in args.tasks:
            gpt4_preference = llm_comparison.gpt4_compare(dataset["question"], answers, baseline_answers)

        # Save answers as json
        with open(join(args.output_dir, "answers.json"), "w") as f:
            json.dump(
                [
                    {
                        "question": q,
                        "answer": a,
                        "baseline_answer": ba,
                    }
                    for q, a, ba in zip(dataset["question"], answers, baseline_answers)
                ],
                f,
            )

        # Print and save report
        report = EvalReport(
            config=args,
            text_complexity=textstat_result.text_standard if textstat_result else None,
            reading_time=textstat_result.reading_time if textstat_result else None,
            toxicity=toxicity_result if toxicity_result else None,
            baseline_text_complexity=baseline_textstat_result.text_standard if baseline_textstat_result else None,
            baseline_reading_time=baseline_textstat_result.reading_time if baseline_textstat_result else None,
            baseline_toxicity=baseline_toxicity_result if baseline_toxicity_result else None,
            gpt4_preference=gpt4_preference if gpt4_preference else None,
            comparison_sheet_path=sheet_path or "",
            data=[
                QuestionAnswerAnswer(question=q, answer=a, baseline_answer=ba)
                for q, a, ba in zip(dataset["question"], answers, baseline_answers)
            ],
        )
        print(report.as_multiline_string())
        report.save(join(args.output_dir, "report.txt"))


    if "knowledge-token" in args.tasks:
        data = ds.load_from_disk("/scratch1/redditqa/cached_datasets/AskHistorians_blank_eval_100.jsonl")
        questions = [q.strip() + " " + a.strip() for q,a in zip(data["question"], data["answer_blank"])]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # greedy decoding
        max_new_tokens = max([len(tokenizer.tokenize(text)) for text in data["expected"]])
        generation_args = {
            "max_new_tokens":max_new_tokens
        }
        wandb.log(dict(
            expected=data["expected"],
            max_new_tokens=max_new_tokens))
        
        # knowledge token filling
        knowledge_answers = _generate_answers(questions, args.model_name, args.adapter_name, generation_args)
        knowledge_token_score = [expected.lower() in generated.lower() for generated, expected in zip(knowledge_answers, data["expected"])].count(True)/len(data["expected"])
        wandb.log(dict(
                knowledge_answers=knowledge_answers, 
                knowledge_token_score=knowledge_token_score))

        # perplexity
        wandb.log(dict(
            perplexity_score=
            calculate_perplexity_for_expected_tokens(questions, data["expected"], args.model_name, args.adapter_name)
            ))

        # knowledge token filling baseline model
        knowledge_answers_baseline = _generate_answers(questions, args.baseline_model_name, None, generation_args)
        knowledge_token_score_baseline = [expected.lower() in generated.lower() for generated, expected in zip(knowledge_answers_baseline, data["expected"])].count(True)/len(data["expected"])
        wandb.log(dict(
                knowledge_answers_baseline=knowledge_answers_baseline, 
                knowledge_token_score_baseline=knowledge_token_score_baseline))

        # perplexity baseline model
        wandb.log(dict(perplexity_score_baseline=
                       calculate_perplexity_for_expected_tokens(questions, data["expected"], args.baseline_model_name, None)
                       ))

if __name__ == "__main__":
    main()
