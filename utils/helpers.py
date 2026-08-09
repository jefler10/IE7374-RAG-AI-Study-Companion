from pathlib import Path

import yaml


CONFIG_FILE = Path("configs/model_config.yaml")


def load_config() -> dict:
    """Load project settings from the YAML configuration file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not config:
        raise ValueError("Configuration file is empty.")

    return config


def ensure_parent_directory(file_path: Path) -> None:
    """Create the parent directory for a file if it does not already exist."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def combine_passage_text(passages: list[dict]) -> str:
    """Combine retrieved passage text into one context string."""
    return "\n\n".join(passage["text"] for passage in passages)
