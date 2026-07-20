# Preliminary Experiments and Initial Results

## Purpose

These preliminary experiments are intended to verify that the selected Retrieval-Augmented Generation approach is feasible before running the full project experiments.

The tests will examine whether the system can:

- Clean and divide biology source text into searchable passages.
- Create embeddings for the passages.
- Retrieve passages relevant to a biology question.
- Generate study materials with FLAN-T5-base.
- Compare instruction-only generation with RAG generation.

## Experimental Environment

The following details will be completed after the first successful run:

- Python version:
- Operating system:
- Processor:
- GPU, if available:
- Generative model: `google/flan-t5-base`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval library: FAISS

## Experiment 1: Data Preprocessing

- Number of source pages:
- Number of processed passages:
- Passage size:
- Passage overlap:
- Result:

## Experiment 2: Retrieval Feasibility

Example query:

> What is the purpose of the cell membrane?

- Number of passages retrieved:
- Top retrieved topic:
- Retrieval score:
- Initial observation:

## Experiment 3: FLAN-T5 Generation

Task tested:

- Summary
- Flashcards
- Multiple-choice question
- Concept explanation

Initial result:

## Experiment 4: Baseline Versus RAG

### Baseline output

The baseline output will be generated using only the written instruction.

### RAG output

The RAG output will be generated using the same instruction plus retrieved textbook passages.

### Initial comparison

This section will be completed after the outputs are generated and reviewed.

## Planned Retrieval Comparison

The project will compare the following retrieval settings:

- `k = 1`
- `k = 3`
- `k = 5`

## Current Limitations

- The initial source collection may be smaller than the final dataset.
- CPU generation may be slow.
- FLAN-T5 has a limited input length.
- Human evaluation results are still preliminary.
- Additional study prompts and biology topics will be added.

## Adjustments Made

This section will document any changes made after testing, such as changes to passage size, overlap, retrieval settings, prompts, or generation parameters.
