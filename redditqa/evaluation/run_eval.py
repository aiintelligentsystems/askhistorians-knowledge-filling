from dataclasses import dataclass, field
from os.path import basename, join
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

from redditqa.data.loader import load_dataset
from redditqa.evaluation import human_comparison, textstat_helper, toxicity
from redditqa.evaluation.util import Result

GENERATION_KWARGS = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "min_length": 32,
    "max_length": 1024,
}


@dataclass
class EvalConfig:
    model_name: Optional[str] = field()
    dataset_name: Optional[str] = field()
    output_dir: Optional[str] = field()
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

Text complexity: {self.text_complexity}
Reading time: {self.reading_time}
Toxicity: {self.toxicity}

Comparison sheet: {self.comparison_sheet_path}

Data:

{data_str}
        """.strip()

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.as_multiline_string())


def main():
    parser = HfArgumentParser(EvalConfig)
    args = parser.parse_args_into_dataclasses()[0]

    # Load the dataset
    dataset = load_dataset(name=args.dataset_name, task="sft").shuffle(42)
    dataset = dataset[args.split]
    dataset = dataset.select(range(args.n_questions))

    # Generate answers
    answers = _generate_answers(dataset["prompt"], args.model_name)

    # Apply metrics
    textstat_result = textstat_helper.calc(answers)
    toxicity_result = toxicity.calc(answers)

    # If baseline model exists, compare with it
    if args.baseline_model_name is not None:
        baseline_answers = _generate_answers(dataset["prompt"], args.baseline_model_name)

        # Create sheet for human comparison
        sheet_path = human_comparison.create_sheet(
            output_path=args.output_dir,
            name=f"Model Comparison {basename(args.model_name)} with {basename(args.baseline_model_name)}",
            questions=dataset["question"],
            answers=answers,
            baseline_answers=baseline_answers,
        )
    else:
        baseline_answers = [""] * len(answers)
        sheet_path = None

    # Print and save report
    report = EvalReport(
        config=args,
        text_complexity=textstat_result.text_standard,
        reading_time=textstat_result.reading_time,
        toxicity=toxicity_result,
        comparison_sheet_path=sheet_path or "",
        data=[
            QuestionAnswerAnswer(question=q, answer=a, baseline_answer=ba)
            for q, a, ba in zip(dataset["question"], answers, baseline_answers)
        ],
    )
    print(report.as_multiline_string())
    report.save(join(args.output_dir, "report.txt"))


def _generate_answers(questions: List[str], model_name: str) -> List[str]:
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate answers
    generation_kwargs = {
        **GENERATION_KWARGS,
        "pad_token_id": tokenizer.pad_token_id,
    }
    answers = pipe(questions, **generation_kwargs)
    answers = [a[0]["generated_text"].replace(q, "").strip() for q, a in zip(questions, answers)]

    return answers


if __name__ == "__main__":
    main()
