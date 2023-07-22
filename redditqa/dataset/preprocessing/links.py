import re
from typing import List

import datasets as ds

regex = r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'


def mask_links(row: str):
    for i, answer in enumerate(row["answers"]):
        row["answers"][i]["answer_body"] = re.sub(regex, "[LINK]", answer["answer_body"])

    return row
