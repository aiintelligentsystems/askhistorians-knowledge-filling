def binary_comparison(answers):
    """Returns tuples of answers, first always best"""
    pairs = []
    
    for i in range(len(answers)-1):
        for j in range(i+1, len(answers)):
            if answers[i]["score"]>answers[j]["score"]:
                pairs.append((answers[i]["body"], answers[j]["body"]))
            elif answers[i]["score"]<answers[j]["score"]:
                pairs.append((answers[j]["body"], answers[i]["body"]))
    return pairs

def preprocess(examples):
    """Returns paired answers (j is better than k). Note that this returns more examples (one for each pair per question)."""
    
    MAX_PAIRS_PER_QUESTION = 10
    n_samples = len(examples["link_id"])
    
    # initialize empty lists for new samples
    new_examples = {"submission_title": [], "response_j": [], "response_k": []}
    for key in examples:
        new_examples[key] = []
    
    for sample_id in range(n_samples):
        # get pairs where first is always the better one
        pairs = binary_comparison(examples["comments"][sample_id])
        
        # sample if we get more pairs than maximum
        if len(pairs) > MAX_PAIRS_PER_QUESTION:
            indices = np.random.choice(list(range(len(pairs))), MAX_PAIRS_PER_QUESTION, replace=False)
            pairs = [pairs[i] for i in indices]
        
        # construct the samples
        for pair in pairs:
            for key in examples:
                new_examples[key].append(examples[key][sample_id])
            new_examples["response_j"].append(pair[0])
            new_examples["response_k"].append(pair[1])
    return new_examples


dataset = ds.load_dataset("json", data_files="/scratch1/jhoff/elif_train.jsonl", split="train", streaming=True)

dataset = dataset.map(preprocess, batch_size=10, batched=True)

# Remove comments column
dataset = dataset.remove_columns(["comments"])

dataset = dataset.shuffle()

next(iter(dataset))