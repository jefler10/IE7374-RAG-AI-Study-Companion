import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_FILE = Path("data/processed/embeddings.npy")
METADATA_FILE = Path("data/processed/passage_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class PassageRetriever:
    """Retrieve biology passages using semantic similarity."""

    def __init__(self) -> None:
        if not EMBEDDING_FILE.exists():
            raise FileNotFoundError(
                "Embeddings were not found. Run embeddings.py first."
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                "Passage metadata was not found. Run embeddings.py first."
            )

        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = np.load(EMBEDDING_FILE).astype("float32")

        with METADATA_FILE.open("r", encoding="utf-8") as metadata_file:
            self.passages = json.load(metadata_file)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the passages most relevant to the query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.passages))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            passage = self.passages[index].copy()
            passage["retrieval_score"] = float(score)
            results.append(passage)

        return results


def main() -> None:
    retriever = PassageRetriever()

    query = "What is the purpose of the cell membrane?"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Score: {result['retrieval_score']:.4f}")
        print(result["text"][:500])


if __name__ == "__main__":
    main()
