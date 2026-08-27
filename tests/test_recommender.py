import pandas as pd

from src.recommender import ItemRecommender


def test_seen_items_are_excluded():
    data = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B", "C", "C"],
        "item_id": ["x", "y", "x", "z", "y", "z"],
        "event_value": [5, 3, 4, 5, 4, 3],
    })
    recs = ItemRecommender().fit(data).recommend("A", n=2)
    assert all(row["item_id"] not in {"x", "y"} for row in recs)

