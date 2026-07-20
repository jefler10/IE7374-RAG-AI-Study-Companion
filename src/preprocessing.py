import json
import re
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/passages.jsonl")

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """Remove repeated whitespace and extra formatting."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if chunk:
            chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


def process_files() -> list[dict]:
    """Read raw text files and create structured passages."""
    passages = []

    for file_path in sorted(RAW_DATA_DIR.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)
        chunks = split_into_chunks(cleaned_text)

        topic = file_path.stem.replace("_", " ").title()

        for index, chunk in enumerate(chunks):
            passages.append(
                {
                    "passage_id": f"{file_path.stem}_{index:03d}",
                    "topic": topic,
                    "source_file": file_path.name,
                    "text": chunk,
                }
            )

    return passages


def save_passages(passages: list[dict]) -> None:
    """Save passages as JSON Lines."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        for passage in passages:
            output_file.write(json.dumps(passage) + "\n")


def main() -> None:
    passages = process_files()

    if not passages:
        raise ValueError(
            "No .txt source files were found in data/raw."
        )

    save_passages(passages)
    print(f"Saved {len(passages)} passages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
