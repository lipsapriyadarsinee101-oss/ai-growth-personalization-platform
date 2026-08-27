import pandas as pd

from src.causal import estimate_randomized_uplift


def test_uplift_direction():
    data = pd.DataFrame({"treatment": [0, 0, 0, 1, 1, 1], "converted": [0, 0, 1, 0, 1, 1]})
    result = estimate_randomized_uplift(data)
    assert result.absolute_uplift > 0

