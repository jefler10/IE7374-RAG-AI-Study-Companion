import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.helpers import load_config


config = load_config()

PATHS_CONFIG = config["paths"]
RETRIEVAL_CONFIG = config["retrieval"]

INPUT_FILE = Path(PATHS_CONFIG["processed_data"])
EMBEDDING_FILE = Path(PATHS_CONFIG["embeddings"])
METADATA_FILE = Path(PATHS_CONFIG["passage_metadata"])

MODEL_NAME = RETRIEVAL_CONFIG["embedding_model"]


def load_passages() -> list:
    """Load processed passages from the JSON Lines file."""
    passages = []

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Processed passages were not found. Run preprocessing.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            passages.append(json.loads(line))

    return passages


def main() -> None:
    passages = load_passages()

    if not passages:
        raise ValueError("No passages were found in the processed data file.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [passage["text"] for passage in passages]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    EMBEDDING_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDING_FILE, embeddings)

    with METADATA_FILE.open("w", encoding="utf-8") as metadata_file:
        json.dump(passages, metadata_file, indent=2)

    print(f"Saved embeddings with shape {embeddings.shape}")
    print(f"Saved passage metadata to {METADATA_FILE}")


if __name__ == "__main__":
    main()
