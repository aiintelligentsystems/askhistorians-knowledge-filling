# RedditQA Project

## Dataset preparation

### Preprocess the dataset

Convert AskHistorians

```
python3 -m redditqa.scripts.pushshift_converter \
    --comments_file /scratch1/redditqa/data/ask_historians/AskHistorians_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/ask_historians/AskHistorians_submissions.jsonl \
    --output_file /scratch1/redditqa/data/ask_historians/AskHistorians.jsonl
```

Convert AskScience

```
python3 -m redditqa.scripts.pushshift_converter \
    --comments_file /scratch1/redditqa/data/ask_science/askscience_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/ask_science/askscience_submissions.jsonl \
    --output_file /scratch1/redditqa/data/ask_science/AskScience.jsonl
```

Convert ELI5

```
python3 -m redditqa.scripts.pushshift_converter \
    --comments_file /scratch1/redditqa/data/eli5/explainlikeimfive_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/eli5/explainlikeimfive_submissions.jsonl \
    --output_file /scratch1/redditqa/data/eli5/explainlikeimfive.jsonl
```

### Create splits

We need to create the train-eval-test splits for our dataset before any filtering.

For ELI5

```
python3 -m redditqa.scripts.create_dataset_split \
    --dataset_file=/scratch1/redditqa/data/eli5/eli5.jsonl \
    --split_file=splits/eli5_split.json
```

## Training

### SFT

### DPO
