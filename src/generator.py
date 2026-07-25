from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-base"


class StudyMaterialGenerator:
    """Generate biology study materials with FLAN-T5."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def generate(
        self,
        instruction: str,
        context=None,
        max_new_tokens: int = 250,
    ) -> str:
        """Generate text with or without retrieved context."""
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if context: 
             prompt = f"""
        You are a biology teaching assistant.

        Use ONLY the information in the context below.

        Follow the task instructions exactly.
        Do not copy sentences from the context.
        Generate a complete answer suitable for an introductory biology student.

        Context:
        {context}

        Task:
        {instruction}

        Response:
        """
        else:
            prompt = instruction

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )


def main() -> None:
    generator = StudyMaterialGenerator()

    sample_context = (
        "The plasma membrane defines the boundary of the cell. "
        "It consists mainly of a phospholipid bilayer with proteins "
        "and controls the movement of substances into and out of the cell."
    )

    output = generator.generate(
        instruction=(
            "Create 3 different flashcards from the context. "
            "Use this exact format:\n"
            "1. Term: ... Definition: ...\n"
            "2. Term: ... Definition: ...\n"
            "3. Term: ... Definition: ..."
        ),
        context=sample_context,
    )

    print(output)


if __name__ == "__main__":
    main()
