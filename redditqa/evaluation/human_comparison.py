import os
from typing import List

import numpy as np
import pandas as pd


def create_sheet(output_path: str, name: str, questions: List[str], answers: List[str], baseline_answers: List[str]):
    n = len(questions)
    # Either 0 or 1 for each question
    order = np.random.randint(2, size=n)

    # Create dataframe
    df = pd.DataFrame(
        {
            "Question": questions,
            "Answer 1": np.where(order == 0, answers, baseline_answers),
            "Answer 2": np.where(order == 0, baseline_answers, answers),
            "Rating": [""] * n,
            "Order": order,
        }
    )
    os.makedirs(output_path, exist_ok=True)
    sheet_path = os.path.join(output_path, f"{name}.xlsx")
    df.to_excel(sheet_path, index=False)
    return sheet_path
