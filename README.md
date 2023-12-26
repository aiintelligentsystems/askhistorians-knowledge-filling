# RedditQA Project

## Dataset preparation

### Preprocess the dataset

Convert AskHistorians

```
python3 -m redditqa.data.pushshift_converter \
    --comments_file /scratch1/redditqa/data/ask_historians/AskHistorians_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/ask_historians/AskHistorians_submissions.jsonl \
    --output_file /scratch1/redditqa/data/ask_historians/AskHistorians.jsonl
```

Convert AskScience

```
python3 -m redditqa.data.pushshift_converter \
    --comments_file /scratch1/redditqa/data/ask_science/askscience_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/ask_science/askscience_submissions.jsonl \
    --output_file /scratch1/redditqa/data/ask_science/AskScience.jsonl
```

Convert ELI5

```
python3 -m redditqa.data.pushshift_converter \
    --comments_file /scratch1/redditqa/data/eli5/explainlikeimfive_comments.jsonl \
    --submissions_file /scratch1/redditqa/data/eli5/explainlikeimfive_submissions.jsonl \
    --output_file /scratch1/redditqa/data/eli5/explainlikeimfive.jsonl
```
