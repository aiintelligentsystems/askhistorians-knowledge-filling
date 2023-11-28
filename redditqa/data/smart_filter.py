import transformers


def question_filter(ds_item: dict, pipeline: transformers.pipeline, verbose=False):
    """
    Returns True if question is graded as good by llm, otherwise False.
    """

    query = """
    You are an expert historian. You curate questions to create a high-quality dataset of history questions. 
    Your goal is to filter out bad questions. You do not have to give explanations for your answer.

    Good questions 

    - cannot be answered in one sentence
    - do not contain a personal reference
    - are not suggestive questions 
    - do not ask for book recommendations
    - do not contain hateful statements

    Here are some examples how to grade questions:
        
    Question: What caused the Wall Street Crash of 1929? 
    Answer: good question

    Question: Wednesday AMA: Magic, Alchemy, and the Occult 
    Answer: bad question

    Question: How did people do math with Roman Numerals?
    Answer: good question

    Question: When does one become a historian?
    Answer: bad question

    Question: How much of a threat was Ivan VI to Catherine the Great's reign as empress?
    Answer: good question

    Question: I need some books on the Asian continent in general.
    Answer: bad question


    Classify the following question into one of the two classes: ['good question', 'bad question']:
    """

    question_title = ds_item['question_title']
    question_text = f"""Question: {question_title}"""
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
    outputs = pipeline(prompt, max_new_tokens=50, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, return_full_text=False)
    output = outputs[0]["generated_text"]
    if verbose:
        print("#" * 50)
        print(question_text)
        print("---" * 30)
        print(output)
        print("#" * 50)

    return output.strip().startswith("Answer: good question")


def answer_filter(ds_item: dict, pipeline: transformers.pipeline, verbose=False):
    """
    Returns dataset item with answers which were graded as good answers by an llm.
    """

    answer_query = """
    You are an expert historian. You curate answers to history questions to create a high-quality dataset of history questions and answers. 
    Your goal is to filter out bad answers. You do not have to give explanations for your answer.

    Good answers 

    - are not yes or no answers
    - do not contain too many personal preferences
    - are not suggestive 
    - do not contain hateful statements
    - can contain references

    Here are some examples how to grade answers:
        
    Question: Did allied soldiers defect to Germany/ The Axis powers during WWII?
    Answer: It was much more common on the Eastern Front, with significant numbers of former Red Army soldiers and civilians from the occupied areas in the east coming over to the German side.  Some of this was a byproduct of an earlier era, with many people in certain regions feeling as though they were occupied by the Soviets to begin with.
    Grade: good answer

    Question: Why did Soviet invade Afghanistan in 1979?
    Answer: If you're a student, wouldn't this perhaps be a good thing to research on your own instead of letting other people do the work for you? Or is this unrelated to school?
    Grade: bad answer

    Question: Do we know of any system of sustained diplomatic relations/interactions between the Hittite Empire and Achaea?
    Answer: We know of a few other contacts between the two states, but not enough to show decisively that there were *sustained* diplomatic relations. In the Tawagalawa letter the Hittite king is clearly treading very carefully, because he wants to avoid offending the Ahhiyawan king, and apologises extensively for earlier aggression (and mentions previous apologies). The Ahhiyawan actions in sheltering a known rebel, Piyamaradu, were not exactly conciliatory. Moreover, we know that Miletos changed hands a couple of times between Ahhiyawa and the Hittites, not necessarily because of wars, but perhaps because of warlords (like Piyamaradu) stirring up trouble. But we know of no treaties or wars as such. So it seems safe to say that the Hittites and Ahhiyawans had an uneasy relationship, but no mortal enmity. They seem to have traded with one another: certainly Mycenaean artefacts (which may or may not be Ahhiyawan) found their way into Anatolia. I'm not aware that trade going the other way was extensive, though. There's a book that very conveniently contains all the documents relating to Ahhiyawa, *The Ahhiyawa Texts*, edited by Beckman, Cline, and Bryce (2012). In it you'll find all the documents relating to Ahhiyawa, including one very tantalising letter from the Ahhiyawan king to the Hittites (but, infuriatingly, it doesn't name the Ahhiyawan king! -- that would have been invaluable for all sorts of reasons, both historical and linguistic). It is not 100% certain that Ahhiyawa = "Achaia", though it is of course very likely. If you read *The Ahhiyawa Texts* you'll see one much later reference to Ahhiyawa in a region much further east, in SE Turkey; it's difficult to decode the implications of this.
    Grade: good answer

    Question: Meaning of Town Suffixes?
    Answer: Sorry, we don't allow [throughout history questions]([LINK]  These tend to produce threads which are collections of trivia, not the in-depth discussions about a particular topic we're looking for.  If you have a specific question about a historical event or period or person, please feel free to re-compose your question and submit it again. Alternatively, questions of this type can be directed to more appropriate subreddits, such as /r/asklinguistics, /r/linguistics (NOTE: do not create a separate post; use the stickied Q&amp;A post), /r/history or /r/askhistory.
    Grade: bad answer

    Classify the following answer into one of the two classes: ['good answer', 'bad answer']:
    """

    question_title = ds_item['question_title']
    answers = ds_item['answers']
    prompts, new_answers = [], []

    for answer in answers:
        question_text = f"""Question: {question_title}\nAnswer: {answer['answer_body']}\n\Please clearly state if this is a good answer or a bad answer, give no explanations."""
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
        prompts.append(prompt)

    outputs = pipeline(prompts, max_new_tokens=50, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, return_full_text=False)

    for answer, output in zip(answers, outputs):
        if verbose:
            print("#" * 50)
            print(question_title)
            print("---" * 30)
            print(answer['answer_body'])
            print("---" * 30)
            print(output[0]["generated_text"].strip())
            print("#" * 50)

        if "good answer" in output[0]["generated_text"].strip()[:100].lower():
            new_answers.append(answer)
    ds_item['answers'] = new_answers
    return ds_item
