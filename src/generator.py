from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.helpers import load_config


class StudyMaterialGenerator:
    """Generate biology study materials with FLAN-T5."""

    def __init__(self) -> None:
        config = load_config()
        generator_config = config["generator"]

        self.model_name = generator_config["model_name"]
        self.max_input_length = generator_config["max_input_length"]
        self.max_new_tokens = generator_config["max_new_tokens"]
        self.num_beams = generator_config["num_beams"]
        self.do_sample = generator_config["do_sample"]
        self.repetition_penalty = generator_config["repetition_penalty"]
        self.no_repeat_ngram_size = generator_config["no_repeat_ngram_size"]
        self.early_stopping = generator_config["early_stopping"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        # Keep the beginning of the prompt when truncation is necessary.
        # Task instructions are placed before the retrieved context.
        self.tokenizer.truncation_side = "right"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        )

    def generate(
        self,
        instruction: str,
        context=None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text with or without retrieved context."""

        if not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if context:
            prompt = f"""
Complete this biology task.

Task:
{instruction}

Use only the biology information below.
Answer the task directly.

Biology information:
{context}

Answer:
"""
        else:
            baseline_instruction = instruction.replace(
                "Use only the provided context.",
                "",
            ).strip()

            prompt = f"""
Complete this biology task.

Task:
{baseline_instruction}

Answer the task directly.

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            early_stopping=self.early_stopping,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )


def main() -> None:
    generator = StudyMaterialGenerator()

    sample_context = (
        "The plasma membrane defines the boundary of the cell. "
        "It consists mainly of a phospholipid bilayer with proteins. "
        "The membrane controls the movement of substances into "
        "and out of the cell."
    )

    output = generator.generate(
        instruction=(
            "Make 3 flashcards from the biology information. "
            "Each flashcard must contain a term and its definition. "
            "Write exactly 3 numbered flashcards. "
            "Example format: "
            "1. Cell - basic unit of life. "
            "2. Nucleus - contains genetic material. "
            "3. Membrane - controls what enters and leaves."
        ),
        context=sample_context,
    )

    print(output)


if __name__ == "__main__":
    main()
