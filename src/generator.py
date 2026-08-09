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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # If the prompt is too long, keep the beginning of the prompt.
        # The task instructions are placed first so they are preserved.
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
You are an introductory biology study assistant.

Complete the task below exactly as requested.
Produce the requested answer itself.
Do not describe the task.
Use complete sentences unless the requested format says otherwise.
Use only information supported by the textbook context.

TASK:
{instruction}

TEXTBOOK CONTEXT:
{context}

ANSWER:
"""
        else:
            # The baseline has no retrieved textbook context, so remove
            # the context-only instruction before sending it to FLAN-T5.
            baseline_instruction = instruction.replace(
                "Use only the provided context.",
                "",
            ).strip()

            prompt = f"""
You are an introductory biology study assistant.

Complete the task below exactly as requested.
Produce the requested answer itself.
Do not describe the task.
Use complete sentences unless the requested format says otherwise.

TASK:
{baseline_instruction}

ANSWER:
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
        "It consists mainly of a phospholipid bilayer with proteins "
        "and controls the movement of substances into and out of the cell."
    )

    output = generator.generate(
        instruction=(
            "Create exactly 3 flashcards from the textbook context.\n\n"
            "Use this exact format:\n"
            "Flashcard 1:\n"
            "Term:\n"
            "Definition:\n\n"
            "Flashcard 2:\n"
            "Term:\n"
            "Definition:\n\n"
            "Flashcard 3:\n"
            "Term:\n"
            "Definition:"
        ),
        context=sample_context,
    )

    print(output)


if __name__ == "__main__":
    main()
