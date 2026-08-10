# IE7374 Personalized Retrieval-Augmented AI Study Companion

## Project Overview

This project develops a Retrieval-Augmented Generation (RAG)-based AI study companion for technical learning. The final proof of concept uses introductory biology material from Biology LibreTexts to generate concise, grounded educational explanations.

For each student query, the system retrieves relevant textbook passages using semantic similarity and provides that context to google/flan-t5-base. The final experiment compares instruction-only generation with RAG using retrieval depths of k=1, k=3, and k=5.

The system is designed as a modular pipeline that could later be adapted to additional subjects by replacing the source material.

---

## Team Members

- Valika Chu
- Jeffrey Lara

---

## Objectives

The primary objectives of this project are to:

- Develop an end-to-end Retrieval-Augmented Generation pipeline for educational use
- Retrieve biology passages relevant to student questions
- Generate concise educational explanations using FLAN-T5-base
- Compare instruction-only generation with retrieval-augmented generation
- Evaluate factual grounding, relevance, readability, completeness, and usefulness
- Investigate how retrieval depth affects response quality
- Build a modular and reproducible pipeline for future expansion
  
---

## Research Questions

This project investigates the following questions:

- Does Retrieval-Augmented Generation improve the factual grounding, relevance, and overall quality of educational explanations compared with instruction-only FLAN-T5-base?
- How does retrieval depth affect response quality when comparing k=1, k=3, and k=5 retrieved passages?

Personalization was part of the original project vision but was not implemented or evaluated in the final proof of concept. It is therefore treated as a possible future extension.

---

## System Architecture

The project follows a Retrieval-Augmented Generation architecture.

### 1. Data Collection

- Collect introductory biology material from Biology LibreTexts
- Use source material with licensing that permits reuse
- Save cleaned source text under `data/raw/`
- Record source and licensing information in `data/attribution.csv`

### 2. Text Preprocessing

- Load raw `.txt` source files
- Normalize whitespace and remove formatting artifacts
- Split each document into overlapping passages
- Assign each passage a unique ID and topic
- Save processed passages in JSON Lines format

### 3. Embedding Generation

- Load the processed passages
- Convert each passage into a semantic embedding using `sentence-transformers/all-MiniLM-L6-v2`
- Normalize embeddings for similarity search
- Save the embedding matrix locally
- Save passage metadata for retrieval

### 4. Semantic Retrieval

- Convert the student question into an embedding
- Compare the query embedding with stored passage embeddings
- Use FAISS to rank passages by semantic similarity
- Retrieve the top-k passages based on the selected retrieval depth
- Return the retrieved passage text, metadata, and similarity scores

### 5. Text Generation

- Load pretrained `google/flan-t5-base`
- Generate an instruction-only baseline response
- Provide the retrieved textbook passages as context
- Generate a RAG response using the instruction and retrieved context
- Save baseline and RAG responses for comparison

### 6. Evaluation

- Compare baseline and RAG outputs
- Evaluate factual grounding
- Evaluate relevance
- Evaluate readability
- Evaluate completeness
- Evaluate usefulness
- Compare performance across `k = 1`, `k = 3`, and `k = 5`
  
---

## Model and Framework Selection

### Generative Model

The primary generative model is google/flan-t5-base.

FLAN-T5 is a pretrained, instruction-tuned encoder-decoder Transformer model designed to follow natural-language instructions across a variety of tasks.

FLAN-T5-base was selected because it provides a practical balance between instruction-following ability and computational requirements. It is large enough to generate useful educational responses while remaining feasible to run within the available project environment.

During development, FLAN-T5-base performed more reliably on concise explanations than on highly structured outputs such as multi-item flashcards or complete multiple-choice questions. For this reason, the final evaluation focused on short biology explanation tasks.

### Generation Strategy

The final pipeline uses greedy decoding for text generation.

Final generation settings are:

- max_input_length = 1024
- max_new_tokens = 250
- num_beams = 1
- do_sample = False
- repetition_penalty = 1.0
- no_repeat_ngram_size = 0
- early_stopping = False

These settings were selected after testing showed that stronger beam-search and repetition-control settings could cause incomplete or poorly structured responses. Greedy decoding with fewer restrictions produced more usable outputs for the final experiment.

Sampling is disabled to reduce generation variability and improve reproducibility across repeated runs.

### Embedding Model

The passage and query embedding model is sentence-transformers/all-MiniLM-L6-v2.

This model converts passages and student questions into 384-dimensional semantic vectors. MiniLM was selected because it provides efficient semantic embeddings with relatively low computational cost, making it suitable for the project’s small-scale retrieval system.

### Retrieval Library

The project uses FAISS for vector-similarity search.

Passage and query embeddings are normalized before retrieval. FAISS then ranks the stored passage embeddings by similarity to the student query and returns the top retrieved passages.

The final experiments compare retrieval depths of:

k = 1
k = 3
k = 5

### Frameworks and Libraries

The implementation uses:

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- NumPy
- pandas
- Beautiful Soup
- PyYAML

---

## Training and Fine-Tuning

This project uses pretrained `google/flan-t5-base` for generation and `sentence-transformers/all-MiniLM-L6-v2` for embeddings. No additional training or fine-tuning was performed.

The final project instead focuses on retrieval-augmented inference and comparison of different retrieval depths while keeping the generative model unchanged.

---

## Dataset

The final dataset uses introductory biology material from Biology LibreTexts under the Creative Commons CC BY 4.0 license.

The repository includes two source files:

    data/raw/cell_membrane.txt
    data/raw/passive_transport.txt

The material covers topics including:

- Cell membrane structure and function
- Fluid mosaic model
- Phospholipids and membrane proteins
- Selective permeability
- Passive transport
- Diffusion
- Facilitated diffusion
- Osmosis and tonicity

Source and licensing information is documented in:

    data/attribution.csv

### Passage Creation

The preprocessing pipeline divides the source text into overlapping passages using:

    Chunk size: 250 words
    Chunk overlap: 50 words

The final processed dataset contains:

    22 passages

Processed passages are saved in:

    data/processed/passages.jsonl

Passage metadata used for retrieval is saved in:

    data/processed/passage_metadata.json

---

## Repository Structure

    IE7374-RAG-AI-Study-Companion/
    │
    ├── configs/
    │   └── model_config.yaml
    │
    ├── data/
    │   ├── raw/
    │   │   ├── cell_membrane.txt
    │   │   └── passive_transport.txt
    │   │
    │   ├── processed/
    │   │   ├── passages.jsonl
    │   │   └── passage_metadata.json
    │   │
    │   ├── attribution.csv
    │   └── evaluation_prompts.json
    │
    ├── docs/
    │   ├── evaluation_rubric.md
    │   ├── method_selection.md
    │
    ├── outputs/
    │   ├── sample_outputs.json
    │   ├── human_evaluation_template.csv
    │   ├── human_evaluation_reviewer1.csv
    │   ├── human_evaluation_reviewer2.csv
    │   ├── human_evaluation_combined_detailed.csv
    │   └── human_evaluation_combined_summary.csv
    │
    ├── src/
    │   ├── __init__.py
    │   ├── embeddings.py
    │   ├── evaluation.py
    │   ├── generator.py
    │   ├── model_runner.py
    │   ├── preprocessing.py
    │   └── retrieval.py
    │
    ├── utils/
    │   ├── __init__.py
    │   └── helpers.py
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

The repository is organized by function:

- `configs/` stores centralized experiment settings
- `data/raw/` stores the cleaned Biology LibreTexts source text
- `data/processed/` stores processed passages and retrieval metadata
- `docs/` stores supporting project documentation
- `outputs/` stores generated responses and final evaluation results
- `src/` contains the preprocessing, embedding, retrieval, generation, experiment, and evaluation code
- `utils/` contains reusable helper functions

### Generated Embedding File

The embedding script creates:

    data/processed/embeddings.npy

This generated file is not committed to GitHub because it can be recreated by running python -m src.preprocessing and python -m src.embeddings.

### Utility Module

The repository includes:

    utils/helpers.py

This module contains reusable helper functions used by the pipeline, including configuration loading from `configs/model_config.yaml`.

The final code uses package-style imports, so commands should be run from the repository root using the module form, for example, python -m src.model_runner.

---

## Installation

Run all commands from the repository root.

### 1. Clone the Repository

```bash
git clone https://github.com/jefler10/IE7374-RAG-AI-Study-Companion.git
cd IE7374-RAG-AI-Study-Companion
```

### 2. Create a Virtual Environment

The project was tested with Python 3.9.7.

Create the environment:

```bash
python3 -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

After activation, the command-line prompt should begin with:

```text
(venv)
```

### 3. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

The initial installation may take several minutes because packages such as PyTorch, Transformers, Sentence Transformers, and FAISS must be installed.

---

## Running the Pipeline

Run all commands from the repository root.

### Step 1: Preprocess the Raw Biology Text

    python -m src.preprocessing

Expected output:

    Saved 22 passages to data/processed/passages.jsonl

This script reads the raw `.txt` files from `data/raw/`, cleans the text, divides the documents into overlapping passages, and saves the processed passages in JSON Lines format.

### Step 2: Create Passage Embeddings

    python -m src.embeddings

Expected output:

    Saved embeddings with shape (22, 384)
    Saved passage metadata to data/processed/passage_metadata.json

This step loads `sentence-transformers/all-MiniLM-L6-v2`, converts each processed passage into a 384-dimensional embedding, and saves the embeddings and passage metadata for retrieval.

The first execution may take longer because the embedding model must be downloaded. Rerun this step whenever the raw source text or preprocessing settings change.

### Step 3: Test Semantic Retrieval

    python -m src.retrieval

This script runs a sample query and displays retrieved passages with their similarity scores.

### Step 4: Test FLAN-T5 Generation

    python -m src.generator

The first execution downloads `google/flan-t5-base`. Generation may take longer when running on a CPU.

### Step 5: Run the Final Baseline and RAG Experiment

    python -m src.model_runner

The final experiment:

- Runs seven biology test prompts
- Generates one instruction-only baseline response for each prompt
- Runs RAG with retrieval depths of `k = 1`, `k = 3`, and `k = 5`
- Saves the retrieved passages and generated responses
- Produces 28 total responses across four experimental conditions
- Saves the results to `outputs/sample_outputs.json`

The retrieval depths are loaded from `configs/model_config.yaml`.

### Step 6: Create the Evaluation Template

    python -m src.evaluation

This script converts the generated experiment outputs into a structured evaluation file containing fields for:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness
- Reviewer name
- Comments

The resulting template is saved to `outputs/human_evaluation_template.csv`.

---

## Reproducing the Experiment

To reproduce the generated experiment outputs from the repository root, run the following commands in order:

    python -m src.preprocessing
    python -m src.embeddings
    python -m src.model_runner
    python -m src.evaluation

The commands must be run in this order because retrieval requires the processed passages and generated embeddings.

The pipeline will:

- Preprocess the two Biology LibreTexts source files
- Create 22 overlapping passages
- Generate 384-dimensional MiniLM embeddings
- Prepare the passage embeddings for FAISS similarity search
- Run seven biology prompts
- Generate one instruction-only baseline response per prompt
- Run RAG with `k = 1`, `k = 3`, and `k = 5`
- Produce 28 total generated responses across four experimental conditions
- Save generated outputs to `outputs/sample_outputs.json`
- Create the evaluation template at `outputs/human_evaluation_template.csv`

The final manual evaluation results used in the project are included in the repository:

- `outputs/human_evaluation_reviewer1.csv`
- `outputs/human_evaluation_reviewer2.csv`
- `outputs/human_evaluation_combined_detailed.csv`
- `outputs/human_evaluation_combined_summary.csv`

These completed evaluation files contain the scores assigned by the two independent evaluators and are not automatically regenerated by the pipeline.

The first run requires internet access to download the pretrained Hugging Face models. Later runs can use locally cached model files.

---
## Representative Test Cases

The final `src/model_runner.py` evaluates seven introductory biology prompts:

1. What is the purpose of the cell membrane?
2. What is the basic structure of the plasma membrane?
3. What does the fluid mosaic model describe?
4. What are membrane proteins?
5. What does selectively permeable mean?
6. What is diffusion?
7. Does passive transport require cellular energy?

Each prompt is evaluated under four conditions:

- Instruction-only baseline
- RAG with `k = 1`
- RAG with `k = 3`
- RAG with `k = 5`

All final prompts request concise one- or two-sentence explanations. This design was selected because FLAN-T5-base was more reliable for short explanatory responses than for highly structured outputs such as multi-item flashcards or complete multiple-choice questions.

Across seven prompts and four experimental conditions, the final experiment produces 28 total responses.

---

## Generated Outputs

Generated files are stored under:

    outputs/

### `outputs/sample_outputs.json`

This file contains the final experiment results for all seven biology prompts.

For each prompt, it includes:

- Task type
- Student query
- Generation instruction
- Instruction-only baseline output
- RAG results for `k = 1`, `k = 3`, and `k = 5`
- Retrieved passages for each retrieval depth
- Generated RAG response for each retrieval depth

Across seven prompts and four experimental conditions, this file contains 28 generated responses.

### `outputs/human_evaluation_template.csv`

This file provides the structured evaluation format used to score generated responses on:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness
- Reviewer name
- Comments

### `outputs/human_evaluation_reviewer1.csv`

This file contains the completed evaluation scores from the first evaluator.

### `outputs/human_evaluation_reviewer2.csv`

This file contains the completed evaluation scores from the second evaluator.

### `outputs/human_evaluation_combined_detailed.csv`

This file combines the detailed evaluation results from both evaluators for all 28 responses.

### `outputs/human_evaluation_combined_summary.csv`

This file contains the final average evaluation scores summarized by experimental condition.

---

## Final Results

The final experiment compared an instruction-only FLAN-T5-base baseline with RAG using retrieval depths of `k = 1`, `k = 3`, and `k = 5`.

Seven biology prompts were evaluated under all four conditions, producing 28 total responses. Two independent evaluators scored each response from 1 to 5 for:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness

The final average scores were:

| Condition | Grounding | Relevance | Readability | Completeness | Usefulness | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.43 | 2.57 | 3.79 | 1.43 | 1.29 | 2.10 |
| RAG k=1 | 3.57 | 3.29 | 4.29 | 2.00 | 2.00 | 3.03 |
| RAG k=3 | 4.21 | 3.86 | 4.21 | 2.86 | 3.21 | 3.67 |
| RAG k=5 | 3.14 | 2.43 | 2.43 | 1.71 | 1.86 | 2.31 |

RAG with `k = 3` achieved the strongest overall performance with an average score of `3.67`. It also produced the highest average scores for factual grounding, relevance, completeness, and usefulness.

RAG with `k = 1` improved substantially over the baseline and achieved the highest readability score of `4.29`.

Increasing retrieval depth to `k = 5` reduced performance. Several outputs became more repetitive, less focused, or less relevant, resulting in an overall score of `2.31`.

A representative example involved passive transport. The instruction-only baseline incorrectly stated that cellular energy was required. In contrast, the RAG `k = 3` response correctly explained that passive transport does not require cellular energy and that substances move from an area of higher concentration to an area of lower concentration.

These results show that retrieval can improve factual grounding and educational usefulness, but they also demonstrate that retrieving more passages does not automatically improve generation quality.

---

## Evaluation Procedure

The final experiment used independent manual evaluation to assess the quality of generated responses.

Two evaluators independently reviewed all 28 outputs produced across the four experimental conditions:

- Instruction-only baseline
- RAG with `k = 1`
- RAG with `k = 3`
- RAG with `k = 5`

Each response was scored from 1 to 5 on the following criteria:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness

Higher scores indicate stronger performance.

The two evaluator scores were combined and averaged to produce the final results summarized in `outputs/human_evaluation_combined_summary.csv`.

Detailed evaluation data is stored in:

- `outputs/human_evaluation_reviewer1.csv`
- `outputs/human_evaluation_reviewer2.csv`
- `outputs/human_evaluation_combined_detailed.csv`
- `outputs/human_evaluation_combined_summary.csv`

The evaluation focused on concise explanatory responses because FLAN-T5-base was more reliable for short explanations than for highly structured outputs such as multi-item flashcards or complete multiple-choice questions.

The detailed scoring criteria are documented in docs/evaluation_rubric.md.

---

## Documentation

Additional project documentation is stored under docs/.

The documentation includes:

- `evaluation_rubric.md` — scoring criteria used to evaluate generated responses
- `method_selection.md` — explanation of model and retrieval-method selection

Final generated outputs and evaluation results are stored under outputs/.

The main README provides the final setup instructions, reproduction steps, experiment description, results, limitations, and project structure.

---

## Configuration

Model, retrieval, preprocessing, generation, and file-path settings are centralized in:

    configs/model_config.yaml

The configuration file is loaded by the pipeline at runtime and controls the main experimental settings used by the project.

Current configuration:

    generator:
      model_name: google/flan-t5-base
      max_input_length: 1024
      max_new_tokens: 250
      num_beams: 1
      do_sample: false
      repetition_penalty: 1.0
      no_repeat_ngram_size: 0
      early_stopping: false

    retrieval:
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
      similarity_metric: cosine
      current_top_k: 1
      planned_top_k_values:
        - 1
        - 3
        - 5

    preprocessing:
      chunk_size_words: 250
      chunk_overlap_words: 50

    paths:
      raw_data: data/raw
      processed_data: data/processed/passages.jsonl
      passage_metadata: data/processed/passage_metadata.json
      embeddings: data/processed/embeddings.npy
      output_file: outputs/sample_outputs.json
      evaluation_template: outputs/human_evaluation_template.csv

The final experiment uses the `planned_top_k_values` setting to evaluate retrieval depths of `k = 1`, `k = 3`, and `k = 5`.

---

## Reproducibility Notes

The final project environment included:

- Python 3.9.7
- macOS
- CPU execution
- PyTorch 2.2.2
- NumPy 1.26.4
- Transformers 4.57.6
- Sentence Transformers
- FAISS CPU
- PyYAML 6.0.3

The workflow was tested inside a Python virtual environment.

The first execution requires internet access to download the pretrained Hugging Face models used by the project:

- `google/flan-t5-base`
- `sentence-transformers/all-MiniLM-L6-v2`

After the models have been downloaded, later runs can use locally cached model files.

FLAN-T5-base can run on a CPU, although generation is slower than it would be on a compatible GPU.

The embedding file `data/processed/embeddings.npy` is generated locally and is not committed to the repository. It can be recreated by running:

    python -m src.preprocessing
    python -m src.embeddings

The final generation configuration uses greedy decoding with sampling disabled to reduce generation variability. Exact outputs may still depend on the software environment, model version, and hardware.

All major experiment settings are controlled through `configs/model_config.yaml`, including:

- Generative model
- Embedding model
- Retrieval depths
- Chunk size
- Chunk overlap
- Generation settings
- File paths

To reproduce the generated experiment outputs, follow the commands listed in the `Reproducing the Experiment` section.

---

## Current Limitations

The final system demonstrates a working proof of concept, but several limitations remain:

- The source material contains only two Biology LibreTexts documents and 22 processed passages
- Topic coverage is limited mainly to cell membranes and passive transport
- The final evaluation uses only seven biology prompts
- The results therefore reflect a small in-domain experiment rather than performance across a full biology curriculum
- FLAN-T5-base is more reliable for concise explanations than for highly structured outputs such as multi-item flashcards or complete multiple-choice questions
- Some responses become repetitive, incomplete, or less focused when more retrieved context is provided
- Retrieval with `k = 5` performed worse than `k = 3`, showing that additional context can introduce noise
- The maximum input length of 1024 tokens may limit how much retrieved context can be used effectively
- CPU inference can be slow
- The embedding index must be rebuilt whenever the source material or chunking settings change
- Personalization was not implemented or evaluated in the final proof of concept
- The evaluation used two independent evaluators, so larger studies with more prompts and evaluators would provide stronger evidence

---

## Troubleshooting

### The Embeddings Are Missing

Run:

    python -m src.preprocessing
    python -m src.embeddings

Then run:

    python -m src.model_runner

### New Source Material Is Not Being Retrieved

Adding or changing a raw text file does not automatically rebuild the embedding index.

Run:

    python -m src.preprocessing
    python -m src.embeddings
    python -m src.model_runner

### The Pipeline Produces Unexpected Results

Confirm that `configs/model_config.yaml` contains the intended settings for:

- Generator model
- Embedding model
- Retrieval depths
- Chunk size
- Chunk overlap
- Generation parameters

Then rerun:

    python -m src.model_runner

### Model Download Takes a Long Time

The first run downloads pretrained models from Hugging Face:

- `google/flan-t5-base`
- `sentence-transformers/all-MiniLM-L6-v2`

Later runs should use locally cached model files.

### CPU Generation Is Slow

FLAN-T5-base can run on a CPU, but generation may take longer than on a compatible GPU. This is expected behavior.

### A Dependency Cannot Be Installed

Install the exact dependencies listed in:

    requirements.txt

The tested environment uses:

    torch==2.2.2
    numpy==1.26.4
    transformers==4.57.6
    PyYAML==6.0.3

If installation problems occur, confirm that the active Python environment is compatible with the versions listed in `requirements.txt`.

---

## References

Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv preprint arXiv:2210.11416.

Johnson, J., Douze, M., and Jégou, H. (2017). *Billion-scale similarity search with GPUs*. arXiv preprint arXiv:1702.08734.

Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33.

Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.
