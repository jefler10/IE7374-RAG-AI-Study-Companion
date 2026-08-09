import csv
import json
from pathlib import Path

from utils.helpers import load_config


config = load_config()
PATHS_CONFIG = config["paths"]

INPUT_FILE = Path(PATHS_CONFIG["output_file"])
OUTPUT_FILE = Path(PATHS_CONFIG["evaluation_template"])


def load_results() -> list:
    """Load generated baseline and multi-k RAG outputs."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Sample outputs were not found. Run model_runner first."
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
        rows.append(
            {
                "sample_id": sample_id,
                "task": result["task"],
                "condition": "baseline",
                "factual_grounding_1_to_5": "",
                "relevance_1_to_5": "",
                "readability_1_to_5": "",
                "completeness_1_to_5": "",
                "usefulness_1_to_5": "",
                "reviewer_name": "",
                "comments": "",
            }
        )

        for rag_key in result["rag_results"]:
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": result["task"],
                    "condition": rag_key,
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
    print(f"Created {len(rows)} evaluation rows.")


if __name__ == "__main__":
    main()
