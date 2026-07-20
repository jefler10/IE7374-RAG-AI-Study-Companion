# Research and Selection of Methods

## Project Objectives

The project will develop a personalized AI study companion for introductory biology. The system will generate summaries, flashcards, multiple-choice questions, concise explanations, and study recommendations.

The main technical objectives are to:

- Build a Retrieval-Augmented Generation pipeline.
- Retrieve relevant biology passages for a student request.
- Generate grounded study materials using a pretrained language model.
- Compare generation with and without retrieval.
- Compare retrieval settings using 1, 3, and 5 passages.
- Explore lightweight topic-level personalization.

## Generative Models Considered

| Model | Advantages | Limitations | Decision |
|---|---|---|---|
| FLAN-T5-small | Fast and computationally efficient | Lower output quality and capacity | Considered for small feasibility tests |
| FLAN-T5-base | Instruction-tuned, suitable for structured educational prompts, and manageable for the project timeline | Slower than FLAN-T5-small | Selected as the primary model |
| GPT-2 | Widely available and easy to load | Not instruction-tuned and less suitable for structured tasks | Not selected |

## Retrieval Methods Considered

| Method | Advantages | Limitations | Decision |
|---|---|---|---|
| Keyword search | Simple and fast | May miss passages with similar meaning but different wording | Not selected as the main method |
| TF-IDF | Useful lexical baseline and computationally efficient | Limited semantic understanding | Possible comparison baseline |
| Sentence embeddings with FAISS | Supports semantic similarity and efficient vector search | Requires embedding generation and index creation | Selected |

## Selected Approach

The project will use `google/flan-t5-base` as the generative model. Biology passages will be converted into sentence embeddings and stored in a FAISS index. When a student submits a request, the system will retrieve the most relevant passages and include them in the FLAN-T5 prompt.

This approach was selected because it supports instruction-based generation while improving factual grounding through retrieved textbook context.

## Framework Selection

The project will use:

- Python for the main implementation.
- Hugging Face Transformers for FLAN-T5 loading and inference.
- PyTorch as the model execution framework.
- Sentence Transformers for passage and query embeddings.
- FAISS for semantic vector search.
- Beautiful Soup for HTML cleaning.
- pandas and JSON for structured data and metadata.

These tools provide pretrained models, established NLP libraries, and manageable computational requirements for the project timeline.

## Computational Considerations

The project will use pretrained models rather than train or fine-tune a model from scratch. This reduces computational cost and allows the team to focus on preprocessing, retrieval, prompt design, generation, and evaluation.

FLAN-T5-base can run on CPU, although generation may be slower than on a GPU. Smaller preliminary tests will be used before running the full experiment set.

## Planned Preliminary Experiments

The team will test:

1. Whether the preprocessing script creates readable overlapping passages.
2. Whether semantic retrieval returns passages relevant to biology questions.
3. Whether FLAN-T5 can generate summaries, flashcards, questions, and explanations.
4. Whether retrieved context improves grounding compared with instruction-only generation.
5. Whether retrieval settings of 1, 3, and 5 passages affect output quality.

## References

- Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Johnson, J., Douze, M., and Jégou, H. (2017). Billion-scale similarity search with GPUs.
