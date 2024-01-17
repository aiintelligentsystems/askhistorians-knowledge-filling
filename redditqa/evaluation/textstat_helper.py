from dataclasses import dataclass
from typing import List

import numpy as np
import textstat

from redditqa.evaluation.util import Result, result_from_values


@dataclass
class TextstatResult:
    text_standard: Result
    reading_time: Result


def calc(answers: List[str]) -> TextstatResult:
    text_standard = [textstat.text_standard(answers, float_output=True) for answers in answers]
    reading_time = [textstat.reading_time(answers) for answers in answers]

    return TextstatResult(
        text_standard=result_from_values(text_standard),
        reading_time=result_from_values(reading_time),
    )
