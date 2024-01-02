import hashlib
import re
from typing import List

import datasets as ds

LINK_REGEX = r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*"

HTML_CHAR_PAIRS = [
    ("&amp;", "&"),
    ("&quot", '"'),
    ("&apos", "'"),
    ("&gt;", ">"),
    ("&lt;", "<"),
]

ELI5_TITLE_STARTS_TO_REMOVE = [
    "LI5",
    "ELI5:",
    "ELI5",
    "ELi5:",
    "ELi5",
    "Eli5:",
    "Eli5",
    "ELIF:",
    "ELIF",
    "eli5",
    "\\[ELI5\\]:",
    "\\[ELI5\\]",
    "\\[eli5\\]:",
    "\\[eli5\\]",
    "ELI5 -",
    "\\(ELI5\\)",
    ": ",
    ", ",
    "- ",
]
ELI5_TITLE_STARTS_TO_REMOVE = [re.compile(f"^{start}") for start in ELI5_TITLE_STARTS_TO_REMOVE]


def create_deterministic_id(row):
    question_title = row["question_title"]
    creation_date = row["question_created_utc"]
    id = hashlib.md5((f"{question_title}{creation_date}".encode('utf-8'))).hexdigest()
    return {"id": id}


def mask_links(row):
    """
    Maps the answers to versions with masked links, i.e. all links are replaced with the string "[LINK]"
    """
    for answer in row["answers"]:
        answer["answer_body"] = _mask_link(answer["answer_body"])
    return row


def _mask_link(s):
    return re.sub(LINK_REGEX, "[LINK]", s)


def replace_html_symbols(row: str):
    """
    Maps the answers to versions with replaced HTML symbols, e.g. "A &amp; B" -> "A & B"
    """
    for answer in row["answers"]:
        answer["answer_body"] = _replace_html_symbols(answer["answer_body"])
    return row


def _replace_html_symbols(s):
    for a, b in HTML_CHAR_PAIRS:
        s = s.replace(a, b)
    return s


def fix_eli5_question_title(row):
    """
    Maps the question title to a version without the ELI5 tag, e.g. "ELI5: Why is the sky blue?" -> "Why is the sky blue?"
    """
    return {"question_title": _preprocess_remove_title_start(row["question_title"])}


def _preprocess_remove_title_start(title):
    for start in ELI5_TITLE_STARTS_TO_REMOVE:
        title = re.sub(start, "", title).strip()
    return title


def get_answer_len_filter(min_len=0, max_len=1e10):
    """
    Returns a filter function that filters out answers that are too short or too long
    """

    def filter_fn(row):
        answers = row["answers"]
        answers = [a for a in answers if min_len <= len(a["answer_body"]) < max_len]
        return {"answers": answers}

    return filter_fn


def remove_answers_with_edit_marker(row):
    """
    Removes answers that contain the string "edit:" in the answer body, e.g.

    "This is a good answer. edit: I added some more information."
    """
    answers = row["answers"]
    answers = [a for a in answers if "edit:" not in a["answer_body"].lower()]
    return {"answers": answers}


def main():
    examples = [
        "https://www.google.com",
        "https://www.wikipedia.org/wiki/Hello#World",
        "https://www.google.com/search?q=hello+world",
        "https://www.nytimes.com/2021/04/30/us/politics/biden-100-days.html",
        "https://www.google.com/maps/place/Hasso+Plattner+Institute+(HPI)/@52.3946672,13.1207002,16.26z/data=!4m6!3m5!1s0x47a85f365d286349:0x1da4e14975e45e72!8m2!3d52.3939965!4d13.1333657!16s%2Fm%2F02qv0ng?entry=ttu",
        "https://www.google.com/maps/place/Hasso+Plattner+Institute+(HPI)/@52.3946672,13.1207002,16.26z/data=!4m6!3m5!1s0x47a85f365d286349:0x1da4e14975e45e72!8m2!3d52.3939965!4d13.1333657!16s%2Fm%2F02qv0ng?entry=ttu",
        "https:ABC",
    ]

    for example in examples:
        print(_mask_link(f"BEGIN {example} END"))

    print("")
    print(_replace_html_symbols("BEGIN &amp; END"))


if __name__ == "__main__":
    main()
