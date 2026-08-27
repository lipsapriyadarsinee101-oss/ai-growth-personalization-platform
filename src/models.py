import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


FEATURES = ["interactions", "unique_items", "engagement_score", "recency_days", "avg_event_value"]


def add_demo_target(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Creates a reproducible proxy label for the synthetic demo only."""
    rng = np.random.default_rng(seed)
    frame = features.copy()
    logits = -1.4 + .08 * frame["recency_days"] - .025 * frame["interactions"] + rng.normal(0, .45, len(frame))
    probability = 1 / (1 + np.exp(-logits))
    frame["churned"] = rng.binomial(1, np.clip(probability, .02, .95))
    return frame


def train_retention_model(frame: pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["churned"], test_size=.25, random_state=42, stratify=frame["churned"]
    )
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, random_state=42).fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return model, auc

