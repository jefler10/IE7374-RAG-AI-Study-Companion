import json
from pathlib import Path

from src.generator import StudyMaterialGenerator
from src.retrieval import PassageRetriever
from utils.helpers import load_config


TEST_CASES = [
    {
        "task": "summary",
        "query": "What is the purpose of the cell membrane?",
        "instruction": (
            "In 1-2 sentences, explain the main purpose of the cell membrane."
        ),
    },
    {
        "task": "structure_explanation",
        "query": "What is the basic structure of the plasma membrane?",
        "instruction": (
            "In 1-2 sentences, describe the basic structure of the plasma membrane."
        ),
    },
    {
        "task": "concept_explanation",
        "query": "What does the fluid mosaic model describe?",
        "instruction": (
            "In 1-2 sentences, explain what the fluid mosaic model describes."
        ),
    },
    {
        "task": "definition",
        "query": "What are membrane proteins?",
        "instruction": (
            "In 1-2 sentences, explain what membrane proteins are and what they do."
        ),
    },
    {
        "task": "concept_explanation",
        "query": "What does selectively permeable mean?",
        "instruction": (
            "In 1-2 sentences, explain what selectively permeable means "
            "for the plasma membrane."
        ),
    },
    {
        "task": "definition",
        "query": "What is diffusion?",
        "instruction": (
            "In 1-2 sentences, explain what diffusion is."
        ),
    },
    {
        "task": "concept_explanation",
        "query": "Does passive transport require cellular energy?",
        "instruction": (
            "In 1-2 sentences, explain whether passive transport requires "
            "cellular energy and how substances move during passive transport."
        ),
    },
]


def combine_context(passages: list) -> str:
    """Combine retrieved passages into one context string."""
    return "\n\n".join(passage["text"] for passage in passages)


def main() -> None:
    """Run baseline and retrieval-depth experiments."""
    config = load_config()

    output_file = Path(config["paths"]["output_file"])
    top_k_values = config["retrieval"]["planned_top_k_values"]

    retriever = PassageRetriever()
    generator = StudyMaterialGenerator()

    results = []

    for test_case in TEST_CASES:
        print(
            f"Running task: {test_case['task']} "
            f"for query: {test_case['query']}"
        )

        baseline_output = generator.generate(
            instruction=test_case["instruction"]
        )

        result = {
            "task": test_case["task"],
            "query": test_case["query"],
            "instruction": test_case["instruction"],
            "baseline_output": baseline_output,
            "rag_results": {},
        }

        for top_k in top_k_values:
            print(f"  Running RAG with k={top_k}")

            retrieved_passages = retriever.retrieve(
                query=test_case["query"],
                top_k=top_k,
            )

            context = combine_context(retrieved_passages)

            rag_output = generator.generate(
                instruction=test_case["instruction"],
                context=context,
            )

            result["rag_results"][f"k_{top_k}"] = {
                "top_k": top_k,
                "retrieved_passages": retrieved_passages,
                "rag_output": rag_output,
            }

        results.append(result)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved {len(results)} experiment results "
        f"to {output_file}"
    )
    print(
        "Retrieval depths tested: "
        + ", ".join(str(k) for k in top_k_values)
    )


if __name__ == "__main__":
    main()
