import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemRecommender:
    def fit(self, events: pd.DataFrame) -> "ItemRecommender":
        matrix = events.pivot_table(index="customer_id", columns="item_id", values="event_value", aggfunc="sum", fill_value=0)
        self.user_item = matrix
        self.similarity = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)
        return self

    def recommend(self, customer_id: str, n: int = 5) -> list[dict]:
        if customer_id not in self.user_item.index:
            raise KeyError(f"Unknown customer: {customer_id}")
        history = self.user_item.loc[customer_id]
        seen = history[history > 0]
        scores = self.similarity[seen.index].mul(seen, axis=1).sum(axis=1)
        scores = scores.drop(index=seen.index, errors="ignore").sort_values(ascending=False).head(n)
        return [{"item_id": item, "score": round(float(score), 4)} for item, score in scores.items()]

