import csv
import json
from pathlib import Path

from utils.helpers import load_config


def create_evaluation_template():
    config = load_config()

    output_file = Path(config["paths"]["output_file"])
    evaluation_file = Path(config["paths"]["evaluation_template"])

    if not output_file.exists():
        raise FileNotFoundError(
            f"Experiment output not found: {output_file}\n"
            "Run python -m src.model_runner first."
        )

    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    sample_id = 1

    for result in results:
        task = result["task"]
        query = result["query"]
        instruction = result["instruction"]

        # Baseline row
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "query": query,
                "instruction": instruction,
                "condition": "baseline",
                "top_k": 0,
                "output_text": result["baseline_output"],
                "factual_grounding": "",
                "relevance": "",
                "readability": "",
                "completeness": "",
                "usefulness": "",
                "reviewer_name": "",
                "comments": "",
            }
        )
        sample_id += 1

        # RAG rows
        for key, rag_result in result["rag_results"].items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "query": query,
                    "instruction": instruction,
                    "condition": key,
                    "top_k": rag_result["top_k"],
                    "output_text": rag_result["rag_output"],
                    "factual_grounding": "",
                    "relevance": "",
                    "readability": "",
                    "completeness": "",
                    "usefulness": "",
                    "reviewer_name": "",
                    "comments": "",
                }
            )
            sample_id += 1

    evaluation_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "task",
        "query",
        "instruction",
        "condition",
        "top_k",
        "output_text",
        "factual_grounding",
        "relevance",
        "readability",
        "completeness",
        "usefulness",
        "reviewer_name",
        "comments",
    ]

    with open(evaluation_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation template to {evaluation_file}")
    print(f"Created {len(rows)} evaluation rows.")


if __name__ == "__main__":
    create_evaluation_template()
