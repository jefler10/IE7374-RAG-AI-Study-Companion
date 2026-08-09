import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.helpers import load_config


class PassageRetriever:
    """Retrieve biology passages using semantic similarity."""

    def __init__(self) -> None:
        config = load_config()

        retrieval_config = config["retrieval"]
        paths_config = config["paths"]

        self.embedding_file = Path(paths_config["embeddings"])
        self.metadata_file = Path(paths_config["passage_metadata"])
        self.model_name = retrieval_config["embedding_model"]
        self.default_top_k = retrieval_config["current_top_k"]

        if not self.embedding_file.exists():
            raise FileNotFoundError(
                "Embeddings were not found. Run embeddings.py first."
            )

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                "Passage metadata was not found. Run embeddings.py first."
            )

        self.model = SentenceTransformer(self.model_name)
        self.embeddings = np.load(
            self.embedding_file
        ).astype("float32")

        with self.metadata_file.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            self.passages = json.load(metadata_file)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def retrieve(
        self,
        query: str,
        top_k=None,
    ) -> list:
        """Return the passages most relevant to the query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = self.default_top_k

        top_k = min(top_k, len(self.passages))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            passage = self.passages[index].copy()
            passage["retrieval_score"] = float(score)
            results.append(passage)

        return results


def main() -> None:
    retriever = PassageRetriever()

    query = "What is the purpose of the cell membrane?"
    results = retriever.retrieve(query)

    print(f"Query: {query}")

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Score: {result['retrieval_score']:.4f}")
        print(result["text"][:500])


if __name__ == "__main__":
    main()
