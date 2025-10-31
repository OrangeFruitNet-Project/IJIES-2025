# evaluation/metrics.py
import numpy as np
from scipy import stats

def mean_sd_ci(values, ci=0.95):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    sd = arr.std(ddof=1) if arr.size > 1 else 0.0
    se = sd / np.sqrt(arr.size) if arr.size>0 else 0.0
    h = se * stats.t.ppf((1 + ci) / 2., arr.size-1) if arr.size>1 else 0.0
    return mean, sd, (mean - h, mean + h)

# Example
# mean, sd, (lower, upper) = mean_sd_ci([96.1, 95.9, 96.3], 0.95)
