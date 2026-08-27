from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class UpliftResult:
    control_rate: float
    treatment_rate: float
    absolute_uplift: float
    relative_uplift: float
    ci_low: float
    ci_high: float
    p_value: float


def estimate_randomized_uplift(data: pd.DataFrame) -> UpliftResult:
    treated = data.loc[data["treatment"] == 1, "converted"].astype(float)
    control = data.loc[data["treatment"] == 0, "converted"].astype(float)
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control groups are required")
    diff = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    _, p_value = stats.ttest_ind(treated, control, equal_var=False)
    relative = diff / control.mean() if control.mean() else np.nan
    return UpliftResult(control.mean(), treated.mean(), diff, relative, ci_low, ci_high, p_value)

