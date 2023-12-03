from dataclasses import dataclass
import transformers
from collections import namedtuple
import torch
import numpy as np

# namedtuple or dataclass not PyArrow serializable

# @dataclass
# class GradedOutput:
#     token_id: torch.Tensor
#     token_str: str
#     probability: float

# graded_output = namedtuple('GradedOutput', 'token_id token_str probability')


def generation_pipeline(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    grading_dict = dict(graded_output=[])

    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # namedtuple or dataclass not PyArrow serializable
        grading_dict['graded_output'].append(dict(token_id=tok.cpu(), token_str=tokenizer.decode(tok), probability=np.exp(score.cpu().numpy())))
    return grading_dict


def question_grading_map(ds_item: dict, pipeline: transformers.pipeline, verbose=False):
    """
    Grades a question with a llm.
    """

    query = """
    You are an expert historian. You curate questions to create a high-quality dataset of history questions. 
    Your goal is to filter out bad questions. You do not have to give explanations for your answer.

    Good questions 

    - should be about an event or person or culture in history
    - may also be about historical method (e.g. “How should we deal with the biases in primary sources?”)
    - do not contain a personal reference
    - are not suggestive questions 
    - do not ask for book recommendations
    - do not contain hateful statements
    - are not poll-type questions (e.g. "Who was the most influential person in history?")

    Here are some examples how to grade questions:

    ***Examples***
    Is the following question a good question (Answer with yes/no)? What caused the Wall Street Crash of 1929? 
    yes

    Is the following question a good question (Answer with yes/no)? Wednesday AMA: Magic, Alchemy, and the Occult 
    no

    Is the following question a good question (Answer with yes/no)? What were the consequences for the British in choosing to hold on to Northern Ireland after World War I?
    yes

    Is the following question a good question (Answer with yes/no)? When does one become a historian?
    no

    Is the following question a good question (Answer with yes/no)? How much of a threat was Ivan VI to Catherine the Great's reign as empress?
    yes

    Is the following question a good question (Answer with yes/no)? I need some books on the Asian continent in general.
    no
    ***Examples end***
    
    Is the following question a good question (Answer with yes/no)? 
    """

    question_title = ds_item['question_title']
    question_text = f"""{question_title}\n"""
    messages = [
        {
            "role": "system",
            "content": query,
        },
        {
            "role": "user", 
            "content": question_text
        },
    ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    generation_out = generation_pipeline(prompt, pipeline.model, pipeline.tokenizer)
    if verbose:
        print(generation_out)
    ds_item.update(generation_out)
    return ds_item

def question_filter(ds_item: dict, accepted_token_str=[], acceptance_thresh=0.2):
    """Filters questions which do not contain `accepted_token_str` in grading with a probability of at least `acceptance_thresh`."""
    output_grading: list[dict] = ds_item['graded_output']
    return any(output["probability"] >= acceptance_thresh for output in output_grading if output["token_str"].lower() in accepted_token_str)

def answer_grading_map(ds_item: dict, pipeline: transformers.pipeline, verbose=False):
    """
    Grades a answer with a llm.
    """

    answer_query = """
    You are an expert historian. You curate answers to history questions to create a high-quality dataset of history questions and answers. 
    Your goal is to filter out bad answers. You do not have to give explanations for your answer.

    Good answers 

    - should be in-depth, comprehensive, accurate, and based off of good quality sources
    - provides the necessary context and complexity that the given topic calls for, going beyond a simple cursory overview
    - do not contain personal anecdotes
    - do not contain suppositions and personal opinions
    - are not suggestive 
    - do not contain hateful statements

    Here are some examples how to grade answers:

    ***Examples start***
    Given the question: Did allied soldiers defect to Germany/ The Axis powers during WWII?
    Is the following answer a good answer (Answer with yes/no)? It was much more common on the Eastern Front, with significant numbers of former Red Army soldiers and civilians from the occupied areas in the east coming over to the German side.  Some of this was a byproduct of an earlier era, with many people in certain regions feeling as though they were occupied by the Soviets to begin with.
    yes

    Given the question: Why did Soviet invade Afghanistan in 1979?
    Is the following answer a good answer (Answer with yes/no)? If you're a student, wouldn't this perhaps be a good thing to research on your own instead of letting other people do the work for you? Or is this unrelated to school?
    no

    Given the question: Meaning of Town Suffixes?
    Is the following answer a good answer (Answer with yes/no)? Sorry, we don't allow [throughout history questions]([LINK]  These tend to produce threads which are collections of trivia, not the in-depth discussions about a particular topic we're looking for.  If you have a specific question about a historical event or period or person, please feel free to re-compose your question and submit it again. Alternatively, questions of this type can be directed to more appropriate subreddits, such as /r/asklinguistics, /r/linguistics (NOTE: do not create a separate post; use the stickied Q&amp;A post), /r/history or /r/askhistory.
    no
    ***Examples end***

    Grade the answers given (Answer with yes/no)
    """

    question_title = ds_item['question_title']
    answers = ds_item['answers']

    for answer in answers:
        question_text = f"""Given the question: {question_title}\nIs the following answer a good answer (Answer with yes/no)? {answer['answer_body']}\n"""
        messages = [
            {
                "role": "system",
                "content": answer_query,
            },
            {
                "role": "user", 
                "content": question_text
            },
        ]
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generation_out = generation_pipeline(prompt, pipeline.model, pipeline.tokenizer)
        if verbose:
            print(generation_out)
        answer.update(generation_out)
    return ds_item

def answer_filter(ds_item: dict, accepted_token_str=[], acceptance_thresh=0.2):
    """Filters answers which do not contain `accepted_token_str` in grading with a probability of at least `acceptance_thresh`."""
    new_answers = []
    for answer in ds_item["answers"]:
        output_grading: list[dict] = answer['graded_output']
        if any(output["probability"] >= acceptance_thresh for output in output_grading if output["token_str"].lower() in accepted_token_str):
            new_answers.append(answer)

    ds_item["answers"] = new_answers
    return ds_item