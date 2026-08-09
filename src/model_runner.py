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
            "Write a short biology summary about the cell membrane. "
            "Explain its main purpose and basic structure in 2-3 sentences."
        ),
    },
    {
        "task": "flashcards",
        "query": "What are the main components of the plasma membrane?",
        "instruction": (
            "Make 3 biology flashcards about the plasma membrane. "
            "Write exactly 3 numbered flashcards. "
            "Each flashcard should contain a term and a short definition."
        ),
    },
    {
        "task": "multiple_choice",
        "query": "How does the phospholipid bilayer help form the membrane?",
        "instruction": (
            "Create 1 biology multiple-choice question about the "
            "phospholipid bilayer. Include 4 answer choices labeled "
            "A, B, C, and D. Then give the correct answer and a "
            "short explanation."
        ),
    },
    {
        "task": "concept_explanation",
        "query": "What does the fluid mosaic model describe?",
        "instruction": (
            "Explain the fluid mosaic model in 2-3 sentences. "
            "Mention the main membrane components and why membrane "
            "fluidity is important."
        ),
    },
    {
        "task": "definition",
        "query": "What are membrane proteins?",
        "instruction": (
            "Explain membrane proteins in 2-3 sentences. "
            "Describe where they are found and what they do."
        ),
    },
    {
        "task": "comparison",
        "query": "How is the plasma membrane selectively permeable?",
        "instruction": (
            "Explain selective permeability in 2-3 sentences. "
            "Describe how the plasma membrane controls what can "
            "enter or leave the cell and why this is important."
        ),
    },
    {
        "task": "concept_explanation",
        "query": (
            "What is diffusion and how does it move substances "
            "across cell membranes?"
        ),
        "instruction": (
            "Explain diffusion in 2-3 sentences. "
            "State the direction substances move, whether cellular "
            "energy is required, and how diffusion relates to the "
            "plasma membrane."
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
