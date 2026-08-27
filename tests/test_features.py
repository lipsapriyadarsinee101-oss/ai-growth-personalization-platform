import pandas as pd

from src.features import customer_features


def test_customer_aggregation():
    events = pd.DataFrame({
        "customer_id": ["A", "A", "B"], "item_id": ["x", "y", "x"],
        "event_value": [1, 3, 5], "event_date": ["2026-01-01", "2026-01-02", "2026-01-02"]
    })
    result = customer_features(events).set_index("customer_id")
    assert result.loc["A", "interactions"] == 2
    assert result.loc["A", "unique_items"] == 2

