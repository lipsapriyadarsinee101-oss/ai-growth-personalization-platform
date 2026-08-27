from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GroundedAssistant:
    """A transparent retrieval baseline; connect an LLM only after retrieval."""
    def __init__(self, knowledge_dir: Path):
        self.paths = sorted(knowledge_dir.glob("*.md"))
        self.documents = [path.read_text(encoding="utf-8") for path in self.paths]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, question: str, top_k: int = 2) -> list[dict]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [{"source": self.paths[i].name, "score": round(float(scores[i]), 3), "content": self.documents[i]} for i in order]

