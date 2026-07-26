from pathlib import Path


def ensure_parent_directory(file_path: Path) -> None:
    """Create the parent directory for a file if it does not already exist."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def combine_passage_text(passages: list[dict]) -> str:
    """Combine retrieved passage text into one context string."""
    return "\n\n".join(passage["text"] for passage in passages)
