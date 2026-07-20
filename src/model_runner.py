import json
from pathlib import Path

from generator import StudyMaterialGenerator
from retrieval import PassageRetriever


OUTPUT_FILE = Path("outputs/sample_outputs.json")


TEST_CASES = [
    {
        "task": "summary",
        "query": "What is the purpose of the cell membrane?",
        "instruction": (
            "Write a concise student-friendly summary explaining "
            "the purpose and structure of the cell membrane."
        ),
    },
    {
        "task": "flashcards",
        "query": "What are the main components of the plasma membrane?",
        "instruction": (
            "Create three term-definition flashcards about the main "
            "components of the plasma membrane."
        ),
    },
    {
        "task": "multiple_choice",
        "query": "How does the phospholipid bilayer help form the membrane?",
        "instruction": (
            "Create one multiple-choice question about the phospholipid "
            "bilayer. Include four answer choices, the correct answer, "
            "and a short explanation."
        ),
    },
    {
        "task": "concept_explanation",
        "query": "What does the fluid mosaic model describe?",
        "instruction": (
            "Explain the fluid mosaic model in clear language for an "
            "introductory biology student."
        ),
    },
]


def combine_context(passages: list[dict]) -> str:
    """Combine retrieved passages into one context string."""
    return "\n\n".join(passage["text"] for passage in passages)


def main() -> None:
    retriever = PassageRetriever()
    generator = StudyMaterialGenerator()

    results = []

    for test_case in TEST_CASES:
        baseline_output = generator.generate(
            instruction=test_case["instruction"]
        )

        retrieved_passages = retriever.retrieve(
            query=test_case["query"],
            top_k=3,
        )

        context = combine_context(retrieved_passages)

        rag_output = generator.generate(
            instruction=test_case["instruction"],
            context=context,
        )

        results.append(
            {
                "task": test_case["task"],
                "query": test_case["query"],
                "instruction": test_case["instruction"],
                "top_k": 3,
                "retrieved_passages": retrieved_passages,
                "baseline_output": baseline_output,
                "rag_output": rag_output,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print(f"Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
