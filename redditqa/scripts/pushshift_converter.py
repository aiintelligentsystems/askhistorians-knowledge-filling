import argparse
import re
from typing import List

import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Disable chained assignment warning.
# See here: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None


SUBMISSION_COLS_TO_KEEP = [
    "id",
    "created_utc",
    "retrieved_on",
    "deleted",
    "title",
    "selftext",
    "score",
]

COMMENT_COLS_TO_KEEP = ["id", "parent_id", "link_id", "created_utc", "retrieved_on", "deleted", "body", "score"]


def convert_pushshift_jsonl_files_to_csv(comments_file: str, submissions_file: str, output_file: str):
    # # Print number of lines
    # print("number of lines")
    # print(f"  submissions: {len(open(submissions_file).readlines())}")
    # print(f"  comments: {len(open(comments_file).readlines())}")

    df_comments, df_submissions = load_comments_and_submission_dfs(comments_file, submissions_file)

    # Delete deleted comments
    df_comments = df_comments.query("deleted == False")

    # Merge submissions and comments
    print("Merging comments and submissions")
    df = merge_dfs(df_comments, df_submissions)
    del df_comments
    del df_submissions

    # Delete deleted submissions
    df = df.query("question_deleted == False")

    # Calculate the length of the texts
    df["question_char_length"] = df["question_title"].apply(len)
    df["question_selftext_char_length"] = df["question_selftext"].apply(len)
    df["answer_char_length"] = df["answer_body"].apply(len)
    df["text_char_length"] = df["question_char_length"] + df["answer_char_length"] + len("Question: \nAnswer:")

    # Fillna for retrieved_on columns as they are missing for some entries
    df["answer_retrieved_on"] = df["answer_retrieved_on"].fillna(0)
    df["question_retrieved_on"] = df["question_retrieved_on"].fillna(0)

    # Nest dataset so that submissions are rows with a column containing a list of comments
    print("Nesting dataset")
    # Get the answers per question
    df_answer_groups = df.reset_index().groupby("submission_id").apply(_group_to_dicts)
    df_answer_groups = df_answer_groups.apply(
        lambda answer_list: [_rename_and_filter_answer_dict(answer) for answer in answer_list]
    )
    df_answer_groups = pd.DataFrame({"answers": df_answer_groups})
    # Create rows per question
    df = df.reset_index()
    question_columns = [col for col in df.columns if col.startswith("question_") or col.startswith("submission_")]
    df = df[question_columns]
    df = df.drop_duplicates()
    # Join questions and answer lists
    df = df.set_index("submission_id").join(df_answer_groups)

    # As a sanity check, let's make sure every submission has comments
    assert df[df.answers.apply(len) == 0].shape[0] == 0, "Some submissions have no comments"

    # Save the dataset
    df = df.reset_index()
    df.to_json(output_file, lines=True, orient="records")
    print(f"Saved dataset to: {output_file}")


def load_comments_and_submission_dfs(comments_file: str, submissions_file: str):
    # Load the comments
    df_comments = pd.read_json(comments_file, lines=True)

    # Convert dates
    df_comments["created_utc"] = pd.to_datetime(df_comments["created_utc"], unit="s", errors="ignore")
    df_comments["retrieved_on"] = pd.to_datetime(df_comments["retrieved_on"], unit="s", errors="ignore")

    # Mark deleted comments
    df_comments["deleted"] = df_comments["body"].apply(lambda x: "[deleted]" in x or "[removed]" in x)

    # # Log info about the comments
    # print("Comments columns")
    # print_col_infos(df_comments)

    # Delete columns we don't need
    print(f"Keeping comment columns: {COMMENT_COLS_TO_KEEP}")
    df_comments = df_comments[COMMENT_COLS_TO_KEEP]

    # Load the submissions
    df_submissions = pd.read_json(submissions_file, lines=True)

    # Mark deleted submissions
    df_submissions["deleted"] = df_submissions["selftext"].apply(lambda x: "[deleted]" in x or "[removed]" in x)

    # Convert dates
    df_submissions["created_utc"] = pd.to_datetime(df_submissions["created_utc"], unit="s", errors="ignore")
    df_submissions["retrieved_on"] = pd.to_datetime(df_submissions["retrieved_on"], unit="s", errors="ignore")

    # # Log info about the submissions
    # print("Submissions columns")
    # print_col_infos(df_submissions)

    # Delete columns we don't need
    print(f"Keeping submission columns: {SUBMISSION_COLS_TO_KEEP}")
    df_submissions = df_submissions[SUBMISSION_COLS_TO_KEEP]

    return df_comments, df_submissions


def print_col_infos(df: pd.DataFrame):
    for col in df.columns:
        percent_na = df[col].isna().mean() * 100
        top_values = df[col].value_counts(sort=True).iloc[:4]
        top_values = ", ".join([f"{str(k)[:20]} ({v})" for k, v in top_values.items()])
        print(f"  {col} [%na={percent_na:.1f}%]: {top_values}")


def merge_dfs(df_comments: pd.DataFrame, df_submissions: pd.DataFrame) -> pd.DataFrame:
    # Keep only top-level comments
    df_comments = df_comments.query("parent_id == link_id")
    df_comments["parent_id"] = df_comments["parent_id"].apply(lambda x: x[3:])
    del df_comments["link_id"]
    print(f"Number of top-level comments: {len(df_comments)}")

    # Create a joint df
    df = df_comments.add_prefix("answer_").join(
        df_submissions.set_index("id").add_prefix("question_"),
        on="answer_parent_id",
        how="inner",  # Only keep entries with both submission and comment
    )

    # Fix name of submission_id
    df = df.rename(columns={"answer_parent_id": "submission_id"})

    # Delete submissions with no comments
    df = df.groupby("submission_id").filter(lambda x: len(x) > 1)

    # Sort by score within each submission
    df = df.sort_values(["submission_id", "answer_score"], ascending=[True, False])

    # Index by answer_link_id and answer_id to show answers per submission
    df = df.set_index(["submission_id", "answer_id"])

    return df


def _group_to_dicts(group):
    return group.to_dict(orient="records")


def _rename_and_filter_answer_dict(answer_dict: dict) -> dict:
    return {k: v for k, v in answer_dict.items() if k.startswith("answer_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments_file", type=str, required=True)
    parser.add_argument("--submissions_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    convert_pushshift_jsonl_files_to_csv(args.comments_file, args.submissions_file, args.output_file)


if __name__ == "__main__":
    main()
