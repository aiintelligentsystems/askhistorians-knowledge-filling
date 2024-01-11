from dataclasses import dataclass
from typing import List

import numpy as np
import textstat


@dataclass
class TextstatResult:
    text_standard_mean: float
    text_standard_std: float
    reading_time_mean: float
    reading_time_std: float


def run_textstat(answers: List[str]) -> TextstatResult:
    text_standard = [textstat.text_standard(answers, float_output=True) for answers in answers]
    reading_time = [textstat.reading_time(answers) for answers in answers]

    result = TextstatResult(
        text_standard_mean=np.mean(text_standard),
        text_standard_std=np.std(text_standard),
        reading_time_mean=np.mean(reading_time),
        reading_time_std=np.std(reading_time),
    )
    return result
