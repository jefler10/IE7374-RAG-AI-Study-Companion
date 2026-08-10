# Research and Selection of Methods

## Project Objectives

The final project develops a Retrieval-Augmented Generation study companion for introductory biology.

The main technical objectives are to:

- Build an end-to-end Retrieval-Augmented Generation pipeline
- Retrieve biology passages relevant to student questions
- Generate concise grounded explanations using a pretrained language model
- Compare instruction-only generation with retrieval-augmented generation
- Compare retrieval depths of `k = 1`, `k = 3`, and `k = 5`
- Evaluate factual grounding, relevance, readability, completeness, and usefulness

Personalization was part of the original project vision but was not implemented or evaluated in the final proof of concept.

## Generative Models Considered

| Model | Advantages | Limitations | Decision |
|---|---|---|---|
| FLAN-T5-small | Fast and computationally efficient | Lower model capacity | Considered during early feasibility testing |
| FLAN-T5-base | Instruction-tuned and manageable within the available computing environment | Slower than FLAN-T5-small and less reliable for highly structured outputs | Selected as the final generative model |
| GPT-2 | Widely available and easy to load | Not instruction-tuned and less suitable for the project’s instruction-based tasks | Not selected |

## Retrieval Methods Considered

| Method | Advantages | Limitations | Decision |
|---|---|---|---|
| Keyword search | Simple and fast | May miss passages with similar meaning but different wording | Not selected |
| TF-IDF | Computationally efficient lexical retrieval | Limited semantic understanding | Considered as an alternative |
| Sentence embeddings with FAISS | Supports semantic similarity and efficient vector search | Requires embedding generation | Selected |

## Selected Approach

The final system uses `google/flan-t5-base` as the generative model and `sentence-transformers/all-MiniLM-L6-v2` for passage and query embeddings.

Biology passages are converted into semantic embeddings and searched using FAISS. For each student question, the system retrieves the top passages according to the selected retrieval depth and provides them to FLAN-T5-base as context.

The final experiment compares:

- Instruction-only baseline
- RAG with `k = 1`
- RAG with `k = 3`
- RAG with `k = 5`

This design allows the effect of retrieval depth to be evaluated while keeping the underlying generative model unchanged.

## Framework Selection

The final implementation uses:

- Python for the main pipeline
- Hugging Face Transformers for FLAN-T5 loading and inference
- PyTorch as the model execution framework
- Sentence Transformers for passage and query embeddings
- FAISS for semantic vector search
- Beautiful Soup for text cleaning support
- pandas and JSON for structured data and evaluation files
- PyYAML for centralized configuration

These tools provide pretrained models, established NLP libraries, and manageable computational requirements for the project environment.

## Computational Considerations

The project uses pretrained models without additional training or fine-tuning.

The final workflow focuses on preprocessing, embedding generation, semantic retrieval, prompt construction, text generation, and evaluation.

FLAN-T5-base was run on CPU during the final experiment. CPU execution is feasible for the project scale, although generation is slower than it would be on a compatible GPU.

## Final Experiment

The final experiment evaluates seven introductory biology prompts across four conditions:

- Instruction-only baseline
- RAG with `k = 1`
- RAG with `k = 3`
- RAG with `k = 5`

This produces 28 generated responses.

The responses are independently evaluated on:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness

The final evaluation showed that RAG with `k = 3` achieved the strongest overall performance.

## References

- Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Johnson, J., Douze, M., and Jégou, H. (2017). Billion-scale similarity search with GPUs.
