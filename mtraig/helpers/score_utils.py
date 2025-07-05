import numpy as np
import pandas as pd
from typing import List

def calculate_faithfulness_score(verifications: List[bool]) -> float:
    if not verifications:
        return 1.0
    ratio = sum(verifications) / len(verifications)
    score = 1 + (ratio * 4)
    return score

def calculate_correlation(df: pd.DataFrame) -> float:
    taus = []
    for example_id, group in df.groupby("example_id"):
        if len(group) < 2:
            continue
        from scipy.stats import pearsonr
        r, _ = pearsonr(group["score_metric"], group["score_human"])
        if not np.isnan(r):
            taus.append(r)
    if not taus:
        return float("nan")
    return float(np.mean(taus)) 