import torch
import torch.nn.functional as F
import numpy as np

from redditqa.evaluation.util import load_model_and_tokenizer

def calculate_perplexity_per_token(input_text: str, model, tokenizer):
    # taken from https://stackoverflow.com/questions/77433100/how-to-get-perplexity-per-token-rather-than-average-perplexity
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to("cuda:0")
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
    per_token_perplexity = torch.exp(loss)
    # sanity check with outputs.loss
    # average_perplexity = torch.exp(torch.mean(loss))
    return per_token_perplexity

def calculate_perplexity_for_expected_tokens(input_texts: str, expected_texts: str, model_name, adapter_name):
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_name)
    perplexities = []
    for input_text, expected_text in zip(input_texts, expected_texts):
        per_token_perplexity = calculate_perplexity_per_token(input_text.strip() + " " + expected_text, model, tokenizer)
        input_text_len = tokenizer(input_text, return_tensors='pt')['input_ids'].size(1) - 1
        # average perplexity over tokens of expected_text
        perplexities.append(per_token_perplexity[input_text_len:].mean().item()) 
    return np.mean(perplexities)
    