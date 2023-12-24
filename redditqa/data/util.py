import re
from typing import List

import datasets as ds

regex = r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*"

html_pairs = [
    ("&amp;", "&"),
    ("&quot", '"'),
    ("&apos", "'"),
    ("&gt;", ">"),
    ("&lt;", "<"),
]


def mask_links(row):
    for answer in row["answers"]:
        answer["answer_body"] = _mask_link(answer["answer_body"])
    return row


def _mask_link(s):
    return re.sub(regex, "[LINK]", s)


def replace_html_symbols(row: str):
    for answer in row["answers"]:
        answer["answer_body"] = _replace_html_symbols(answer["answer_body"])
    return row


def _replace_html_symbols(s):
    for a, b in html_pairs:
        s = s.replace(a, b)
    return s


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
