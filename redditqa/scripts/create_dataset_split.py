import argparse
import json

from redditqa.data.huggingface_dataset import load_reddit_dataset
from redditqa.data.util import create_deterministic_id

SEED = 42


def main():
    """
    Create a dataset split for a reddit dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--test_pct", type=float, default=0.15)
    parser.add_argument("--eval_pct", type=float, default=0.15)
    args = parser.parse_args()

    # Load the dataset (already has ids)
    dataset = load_reddit_dataset(args.dataset_file)
    dataset = dataset.shuffle(seed=SEED)

    # Create indices for the splits
    train_pct = 1 - args.test_pct - args.eval_pct
    eval_pct = args.eval_pct
    train_end_index = int(len(dataset) * train_pct)
    eval_end_index = train_end_index + int(len(dataset) * eval_pct)
    print(f"Train ends at {train_end_index}. Eval ends at {eval_end_index}")

    # Create the dataset splits
    dataset_train = dataset.select(range(0, train_end_index))
    dataset_eval = dataset.select(range(train_end_index, eval_end_index))
    dataset_test = dataset.select(range(eval_end_index, len(dataset)))

    # Log the sizes
    print(f"Train size: {len(dataset_train)}")
    print(f"Eval size: {len(dataset_eval)}")
    print(f"Test size: {len(dataset_test)}")

    # Create splits
    selection_dict = {
        "train": dataset_train["id"],
        "eval": dataset_eval["id"],
        "test": dataset_test["id"],
    }

    # Save the splits
    with open(args.split_file, "w") as f:
        json.dump(selection_dict, f, indent=4)
    print(f"Saved splits to {args.split_file}")


if __name__ == "__main__":
    main()
