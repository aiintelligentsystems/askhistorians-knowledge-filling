from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

from redditqa.data.loader import load_dataset
from redditqa.evaluation import textstat_helper, toxicity
from redditqa.evaluation.util import Result


@dataclass
class EvalConfig:
    model_name: Optional[str] = field()
    dataset_name: Optional[str] = field()
    output_path: Optional[str] = field()
    split: Optional[str] = field(default="test")
    n_questions: Optional[int] = field(default=100)


GENERATION_KWARGS = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "min_length": 32,
    "max_length": 128,
}


@dataclass
class EvalReport:
    config: EvalConfig
    text_complexity: Result
    reading_time: Result
    toxicity: Result

    def as_multiline_string(self):
        return f"""\
        Model: {self.config.model_name}
        Dataset: {self.config.dataset_name}
        Split: {self.config.split}
        N questions: {self.config.n_questions}
        
        Text complexity: {self.text_complexity}
        Reading time: {self.reading_time}
        Toxicity: {self.toxicity}
        """

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

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Generate answers
    answers = _generate_answers(dataset["prompt"], model, tokenizer)

    # Apply metrics
    textstat_result = textstat_helper.run_textstat(answers)
    toxicity_result = toxicity.run_classifier(answers)

    # Print and save report
    report = EvalReport(
        config=args,
        text_complexity=textstat_result.text_standard,
        reading_time=textstat_result.reading_time,
        toxicity=toxicity_result,
    )
    print(report.as_multiline_string())
    report.save(args.output_path)


def _generate_answers(questions, model, tokenizer):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    generation_kwargs = {
        **GENERATION_KWARGS,
        "pad_token_id": tokenizer.pad_token_id,
    }
    answers = pipe(questions, **generation_kwargs)

    return answers


if __name__ == "__main__":
    main()
