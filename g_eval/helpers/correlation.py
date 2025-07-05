"""
Correlation calculation utility for G-Eval detection.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def calculate_correlation(df: pd.DataFrame) -> float:
    """
    Compute the "instance-level" Pearson correlation as in the LFTQA-Eval paper.
    """
    taus = []
    for example_id, group in df.groupby("example_id"):
        if len(group) < 2:
            continue
        r, _ = pearsonr(group["score_metric"], group["score_human"])
        if not np.isnan(r):
            taus.append(r)
    return float(np.mean(taus)) if taus else float("nan") 