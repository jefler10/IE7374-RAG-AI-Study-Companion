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
        max_new_tokens: int = 120,
    ) -> str:
        """Generate text with or without retrieved context."""
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if context:
            prompt = (
                "Use only the biology textbook context below to complete "
                "the instruction.\n\n"
                f"Context:\n{context}\n\n"
                f"Instruction:\n{instruction}"
            )
        else:
            prompt = instruction

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
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
