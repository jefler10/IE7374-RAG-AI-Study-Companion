# Preliminary Experiments and Initial Results

## Purpose

These preliminary experiments verified that the selected Retrieval-Augmented Generation approach is feasible before the full project experiments are completed.

The tests examined whether the system could:

- Clean and divide biology source text into searchable passages.
- Create embeddings for the passages.
- Retrieve passages relevant to a biology question.
- Generate study materials with FLAN-T5-base.
- Compare instruction-only generation with RAG generation.

## Experimental Environment

- Python version: 3.9.7
- Operating system: macOS
- Processor: Intel-based Mac
- GPU, if available: None used
- Generative model: `google/flan-t5-base`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval library: FAISS

## Experiment 1: Data Preprocessing

- Number of source pages: 1
- Number of processed passages: 3
- Passage size: 250 words
- Passage overlap: 50 words
- Result: The preprocessing script successfully cleaned the source text and saved three overlapping passages to `data/processed/passages.jsonl`.

## Experiment 2: Retrieval Feasibility

Example query:

> What is the purpose of the cell membrane?

- Number of passages retrieved: 3
- Top retrieved topic: Cell membrane
- Top retrieval score: 0.5188
- Initial observation: All three retrieved passages were relevant to the cell membrane. The highest-ranked passage discussed membrane proteins and transport, while another passage directly described the membrane as the boundary of the cell.

## Experiment 3: FLAN-T5 Generation

Tasks tested:

- Summary
- Flashcards
- Multiple-choice question
- Concept explanation

Initial result: FLAN-T5-base loaded successfully and generated biology text using both instruction-only prompts and retrieved textbook context. Some prompt refinement was needed to reduce repetition and improve output formatting.

## Experiment 4: Baseline Versus RAG

### Baseline output

The system successfully generated outputs using only the written instruction.

### RAG output

The system successfully generated outputs using the same instruction plus three retrieved textbook passages.

### Initial comparison

Four baseline and RAG comparisons were generated and saved to `outputs/sample_outputs.json`. A human-evaluation template was also created for scoring factual grounding, relevance, readability, completeness, and usefulness.

## Planned Retrieval Comparison

The project will compare:

- `k = 1`
- `k = 3`
- `k = 5`

## Current Limitations

- The current feasibility dataset contains only one source page.
- CPU generation may be slow.
- FLAN-T5 has a limited input length.
- Human evaluation results are still preliminary.
- Additional study prompts and biology topics will be added.
- Lightweight personalization has not yet been fully implemented.

## Adjustments Made

- Changed PyTorch from version 2.8.0 to 2.2.2 for compatibility with Python 3.9.7.
- Changed NumPy from version 2.0.2 to 1.26.4 for compatibility.
- Replaced the Python 3.10 type syntax `str | None` with `context=None`.
- Added repetition controls to FLAN-T5 generation.
- Tested the complete pipeline in a fresh virtual environment.
