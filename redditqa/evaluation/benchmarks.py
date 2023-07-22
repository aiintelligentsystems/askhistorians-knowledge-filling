from lm_eval import evaluator

from redditqa.utils import prefix_dict

tasks = [
    "openbookqa",
    "truthfulqa_gen",
    "truthfulqa_mc",
    "triviaqa",
]


def benchmark(model_path: str):
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=f"pretrained='{model_path}',load_in_8bit=True,device_map_option='auto',use_accelerate=True",
        tasks=tasks,
        batch_size=1,
        max_batch_size=1,
        device="cuda",
    )

    # Prefix the task name to the metric name
    results = {task: prefix_dict(results[task], task) for task in tasks}
    return results
