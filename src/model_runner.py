import json
from pathlib import Path

from generator import StudyMaterialGenerator
from retrieval import PassageRetriever
from utils.helpers import load_config


TEST_CASES = [
    {
        "task": "summary",
        "query": "What is the purpose of the cell membrane?",
        "instruction": """
Write a concise student-friendly summary explaining the purpose
and structure of the cell membrane.

Use only the provided context.
""",
    },
    {
        "task": "flashcards",
        "query": "What are the main components of the plasma membrane?",
        "instruction": """
Create exactly three flashcards.

Format each flashcard as:

Flashcard 1:
Term:
Definition:

Flashcard 2:
Term:
Definition:

Flashcard 3:
Term:
Definition:

Use only the provided context.
""",
    },
    {
        "task": "multiple_choice",
        "query": "How does the phospholipid bilayer help form the membrane?",
        "instruction": """
Create one multiple-choice question.

Format:

Question:

A.
B.
C.
D.

Correct Answer:

Explanation:

Use only the provided context.
""",
    },
    {
        "task": "concept_explanation",
        "query": "What does the fluid mosaic model describe?",
        "instruction": """
Explain the fluid mosaic model.

Include:

- What the model describes
- The major components involved
- Why membrane fluidity is important

Write 3-5 sentences.

Use only the provided context.
""",
    },
    {
        "task": "definition",
        "query": "What are membrane proteins?",
        "instruction": """
Explain the role of membrane proteins.

Include:

- What membrane proteins are
- Where they are located
- Their function in the plasma membrane

Use only the provided context.
""",
    },
    {
        "task": "comparison",
        "query": "How is the plasma membrane selectively permeable?",
        "instruction": """
Explain selective permeability.

Include:

- What selective permeability means
- How the membrane controls the movement of materials
- Why selective permeability is important for cells

Write 3-5 sentences.

Use only the provided context.
""",
    },
    {
        "task": "concept_explanation",
        "query": (
            "What is diffusion and how does it move substances "
            "across cell membranes?"
        ),
        "instruction": """
Explain diffusion in clear language for an introductory biology student.

Include:

- What diffusion means
- The direction substances move
- Whether diffusion requires cellular energy
- How diffusion relates to the plasma membrane

Write 3-5 sentences.

Use only the provided context.
""",
    },
]


def combine_context(passages: list) -> str:
    """Combine retrieved passages into one context string."""
    return "\n\n".join(passage["text"] for passage in passages)


def main() -> None:
    """Run baseline and retrieval-augmented generation test cases."""
    config = load_config()

    output_file = Path(config["paths"]["output_file"])
    top_k = config["retrieval"]["current_top_k"]

    retriever = PassageRetriever()
    generator = StudyMaterialGenerator()

    results = []

    for test_case in TEST_CASES:
        baseline_output = generator.generate(
            instruction=test_case["instruction"]
        )

        retrieved_passages = retriever.retrieve(
            query=test_case["query"],
            top_k=top_k,
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
                "top_k": top_k,
                "retrieved_passages": retrieved_passages,
                "baseline_output": baseline_output,
                "rag_output": rag_output,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(results)} results to {output_file}")
    print(f"Retrieval top_k used: {top_k}")


if __name__ == "__main__":
    main()
