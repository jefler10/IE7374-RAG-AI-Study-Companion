import csv
import json
from pathlib import Path

from utils.helpers import load_config


config = load_config()
PATHS_CONFIG = config["paths"]

INPUT_FILE = Path(PATHS_CONFIG["output_file"])
OUTPUT_FILE = Path(PATHS_CONFIG["evaluation_template"])


def load_results() -> list:
    """Load generated baseline and RAG outputs."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Sample outputs were not found. Run model_runner.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def main() -> None:
    results = load_results()

    fieldnames = [
        "sample_id",
        "task",
        "condition",
        "factual_grounding_1_to_5",
        "relevance_1_to_5",
        "readability_1_to_5",
        "completeness_1_to_5",
        "usefulness_1_to_5",
        "reviewer_name",
        "comments",
    ]

    rows = []

    for sample_id, result in enumerate(results, start=1):
        for condition in ["baseline_output", "rag_output"]:
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": result["task"],
                    "condition": condition,
                    "factual_grounding_1_to_5": "",
                    "relevance_1_to_5": "",
                    "readability_1_to_5": "",
                    "completeness_1_to_5": "",
                    "usefulness_1_to_5": "",
                    "reviewer_name": "",
                    "comments": "",
                }
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation template to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
