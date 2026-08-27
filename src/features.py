import pandas as pd


def customer_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data into leakage-safe customer features."""
    frame = events.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    observation_end = frame["event_date"].max() + pd.Timedelta(days=1)
    grouped = frame.groupby("customer_id")
    result = grouped.agg(
        interactions=("item_id", "size"),
        unique_items=("item_id", "nunique"),
        engagement_score=("event_value", "sum"),
        last_event=("event_date", "max"),
    ).reset_index()
    result["recency_days"] = (observation_end - result.pop("last_event")).dt.days
    result["avg_event_value"] = result["engagement_score"] / result["interactions"]
    return result

