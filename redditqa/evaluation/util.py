from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Result:
    mean: float
    std: float

    def __str__(self):
        return f"{self.mean:.2f} ± {self.std:.2f}"


def result_from_values(values: List[float]) -> Result:
    return Result(mean=np.mean(values), std=np.std(values))
