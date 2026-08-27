from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"


def main(seed: int = 42, customers: int = 1200, events: int = 18000) -> None:
    rng = np.random.default_rng(seed)
    OUT.mkdir(exist_ok=True)
    customer_ids = np.array([f"C{i:05d}" for i in range(customers)])
    item_ids = np.array([f"ITEM_{i:03d}" for i in range(80)])
    segments = rng.choice(["new", "engaged", "at_risk", "premium"], customers, p=[.25, .40, .20, .15])
    customer_segment = dict(zip(customer_ids, segments))
    picked_customers = rng.choice(customer_ids, events)
    event_types = rng.choice(["view", "play", "purchase"], events, p=[.62, .28, .10])
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180, events), unit="D")
    event_df = pd.DataFrame({
        "customer_id": picked_customers,
        "item_id": rng.choice(item_ids, events),
        "event_type": event_types,
        "event_value": np.select([event_types == "view", event_types == "play"], [1, 3], default=5),
        "event_date": dates,
    })
    event_df.to_csv(OUT / "events.csv", index=False)

    treatment = rng.integers(0, 2, customers)
    base = np.array([{"new": .13, "engaged": .31, "at_risk": .08, "premium": .42}[customer_segment[c]] for c in customer_ids])
    conversion_probability = np.clip(base + treatment * .055, 0, 1)
    experiment_df = pd.DataFrame({
        "customer_id": customer_ids,
        "segment": [customer_segment[c] for c in customer_ids],
        "treatment": treatment,
        "converted": rng.binomial(1, conversion_probability),
    })
    experiment_df.to_csv(OUT / "experiment.csv", index=False)
    print(f"Created {len(event_df):,} events and {len(experiment_df):,} experiment rows in {OUT}")


if __name__ == "__main__":
    main()

