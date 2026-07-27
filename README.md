# IE7374 Personalized Retrieval-Augmented AI Study Companion

## Project Overview

This project develops a personalized AI study companion using Retrieval-Augmented Generation (RAG) for technical learning.

The current implementation uses introductory biology textbook material to generate educational study resources, including:

- Student-friendly summaries
- Flashcards
- Multiple-choice questions
- Definitions
- Concept explanations
- Topic comparisons

The system combines semantic passage retrieval with a pretrained generative language model. For each student query, the pipeline retrieves a relevant textbook passage and provides that passage to the model as supporting context.

The system produces two types of responses:

1. A **baseline output** generated from the instruction without retrieved context
2. A **RAG output** generated from the same instruction with a retrieved textbook passage


---

## Team Members

- Valika Chu
- Jeffrey Lara

---

## Objectives

The primary objectives of this project are to:

- Develop an AI-powered study assistant for technical learning
- Implement an end-to-end Retrieval-Augmented Generation pipeline
- Retrieve biology passages relevant to student questions
- Generate summaries, flashcards, multiple-choice questions, definitions, and explanations
- Compare baseline generation with retrieval-augmented generation
- Evaluate factual grounding, relevance, readability, completeness, and usefulness
- Investigate how different retrieval settings affect output quality
- Explore lightweight personalization based on topic-level student performance

---

## Research Questions

This project investigates the following questions:

1. How does RAG affect the factual grounding and relevance of FLAN-T5-generated study materials compared with generation without retrieved textbook passages?

2. How does the number of retrieved passages affect output quality when using `k = 1`, `k = 3`, and `k = 5`?

3. Can lightweight topic-level performance tracking improve the usefulness of practice questions and study recommendations?

The current pipeline uses:

```text
top_k = 1
```

Experiments comparing `k = 1`, `k = 3`, and `k = 5` are planned for the final project stage.

---

## System Architecture

The project follows a Retrieval-Augmented Generation architecture.

### 1. Data Collection

- Collect introductory biology material from Biology LibreTexts
- Use pages with licensing that permits reuse
- Save cleaned source text under `data/raw/`
- Record source and licensing information in `data/attribution.csv`

### 2. Text Preprocessing

- Load raw `.txt` source files
- Normalize whitespace and remove formatting artifacts
- Split each document into overlapping passages
- Assign a unique passage ID and topic
- Save processed passages in JSON Lines format

### 3. Embedding Generation

- Load the processed passages
- Convert each passage into a semantic embedding
- Use `sentence-transformers/all-MiniLM-L6-v2`
- Normalize embeddings for similarity search
- Save the embedding matrix locally
- Save passage metadata for retrieval

### 4. Semantic Retrieval

- Convert the student query into an embedding
- Compare the query embedding with passage embeddings
- Use FAISS to retrieve the most relevant passage
- Return the passage text, source information, and similarity score

### 5. Text Generation

- Load pretrained `google/flan-t5-base`
- Generate an instruction-only baseline response
- Retrieve a relevant textbook passage
- Generate a RAG response using the instruction and retrieved context
- Save both responses for comparison

### 6. Evaluation

- Compare baseline and RAG outputs
- Review factual grounding
- Review relevance
- Review readability
- Review completeness
- Review usefulness
- Check whether structured outputs follow the requested format

---

## Model and Framework Selection

### Generative Model

The primary generative model is:

```text
google/flan-t5-base
```

FLAN-T5 is a pretrained, instruction-tuned encoder-decoder Transformer model. It can respond to natural-language instructions, making it suitable for tasks such as:

- Writing summaries
- Creating flashcards
- Generating multiple-choice questions
- Explaining biology concepts
- Defining technical terms

FLAN-T5-base was selected because it provides more generation capacity than FLAN-T5-small while remaining practical for the project timeline and available computing resources.

### Generation Strategy

The current pipeline uses deterministic beam search for text generation.

Current generation settings are:

```text
max_input_length = 1024
max_new_tokens = 250
num_beams = 4
do_sample = False
repetition_penalty = 1.2
no_repeat_ngram_size = 3
early_stopping = True
```

Beam search was selected to improve reproducibility and consistency for structured educational tasks such as flashcards, multiple-choice questions, definitions, and concept explanations.

Because random sampling is disabled, repeated runs with the same prompts, model version, and retrieved passages should produce consistent outputs.

The current results show that deterministic beam search improves reproducibility, but it does not guarantee that FLAN-T5 will follow every requested format or produce complete answers.

### Embedding Model

The passage and query embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts passages and queries into compact semantic vectors.

The embedding dimension is:

```text
384
```

### Retrieval Library

The project uses FAISS for vector-similarity search.

Passage and query embeddings are normalized before retrieval. Inner-product search on normalized vectors functions as cosine-similarity search.

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

This project does not train a language model from scratch and does not fine-tune FLAN-T5 during Milestone 4.

The system uses:

- Pretrained `google/flan-t5-base` for text-generation inference
- Pretrained `sentence-transformers/all-MiniLM-L6-v2` for passage and query embeddings

Because no supervised fine-tuning is performed, conventional training, validation, and test splits are not currently required.

Instead, the project uses representative evaluation prompts to test retrieval and generation behavior.

The current work focuses on:

- Source collection and attribution
- Text cleaning and passage creation
- Embedding generation
- Semantic retrieval
- Prompt construction
- Baseline generation
- RAG generation
- Preliminary human evaluation

---

## Dataset

The current dataset contains cleaned introductory biology material from Biology LibreTexts.

The repository includes two raw source files:

```text
data/raw/cell_membrane.txt
data/raw/passive_transport.txt
```

The current material covers:

- Cell membrane structure and function
- The fluid mosaic model
- Phospholipids
- Membrane proteins
- Selective permeability
- Passive transport
- Diffusion
- Facilitated diffusion
- Osmosis
- Tonicity

Source and licensing information is documented in:

```text
data/attribution.csv
```

The attribution file contains information for both source pages, including:

- Page identifier
- Topic
- Page title
- Author or contributors
- Source URL
- License
- Access date

### Passage Creation

The preprocessing pipeline divides the raw source text into overlapping passages.

Current settings:

```text
Chunk size: 250 words
Chunk overlap: 50 words
```

The current processed dataset contains:

```text
22 passages
```

The passages are saved in:

```text
data/processed/passages.jsonl
```

The dataset will be expanded during the final project stage to include additional introductory biology topics.

---

## Repository Structure

```text
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
│   └── preliminary_results.md
│
├── outputs/
│   ├── README.md
│   ├── human_evaluation_template.csv
│   └── sample_outputs.json
│
├── src/
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generator.py
│   ├── model_runner.py
│   ├── preprocessing.py
│   └── retrieval.py
│
├── utils/
│   └── helpers.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Generated Embedding File

The embedding script creates:

```text
data/processed/embeddings.npy
```

This generated file is not committed to GitHub because it can be recreated by running:

```bash
python3 src/embeddings.py
```

### Utility Module

The repository includes:

```text
utils/helpers.py
```

This file provides reusable helper functions that can be used across pipeline components.

The current `src/model_runner.py` still contains its own local `combine_context()` helper function and does not currently import `combine_passage_text()` from `utils/helpers.py`.

The utility module is therefore present as reusable shared logic, but full integration into the current runner remains future cleanup work.

This design choice preserves compatibility with the required command:

```bash
python src/model_runner.py
```

without requiring changes to Python package import paths.

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

```bash
python3 src/preprocessing.py
```

Expected output:

```text
Saved 22 passages to data/processed/passages.jsonl
```

This script reads all `.txt` files from:

```text
data/raw/
```

It cleans the text and divides it into overlapping passages.

### Step 2: Create Passage Embeddings

```bash
python3 src/embeddings.py
```

Expected output:

```text
Saved embeddings with shape (22, 384)
Saved passage metadata to data/processed/passage_metadata.json
```

The first execution downloads the sentence-transformer model.

This step must be rerun whenever:

- A new source file is added
- `passages.jsonl` changes
- Passage chunking settings change
- Existing source text is modified

### Step 3: Test Semantic Retrieval

```bash
python3 src/retrieval.py
```

This script runs a sample query and displays retrieved passages with their similarity scores.

### Step 4: Test FLAN-T5 Generation

```bash
python3 src/generator.py
```

The first execution downloads:

```text
google/flan-t5-base
```

Generation may take longer when running on a CPU.

### Step 5: Run the Complete Baseline and RAG Pipeline

The main Milestone 4 command is:

```bash
python3 src/model_runner.py
```

The assignment command may also be used when the environment maps `python` to Python 3:

```bash
python src/model_runner.py
```

The script:

- Loads the processed passage data
- Loads passage metadata and embeddings
- Loads pretrained FLAN-T5-base
- Runs seven representative test cases
- Generates an instruction-only baseline for each test case
- Retrieves one passage for each query
- Generates a RAG response using the retrieved passage
- Saves the results under `outputs/`

Expected output:

```text
Saved 7 results to outputs/sample_outputs.json
```

### Step 6: Create the Human-Evaluation Template

```bash
python3 src/evaluation.py
```

Expected output:

```text
Saved evaluation template to outputs/human_evaluation_template.csv
```

---

## Reproducing the Current Results

To recreate all processed data, embeddings, generated outputs, and evaluation files, run:

```bash
python3 src/preprocessing.py
python3 src/embeddings.py
python3 src/model_runner.py
python3 src/evaluation.py
```

The commands must be run in this order because retrieval requires processed passages and generated embeddings.

After preprocessing and embedding generation are complete, the main inference pipeline can be rerun using:

```bash
python3 src/model_runner.py
```

Because deterministic beam search is used, rerunning the same model version with the same inputs should produce consistent generation results.

---

## Representative Test Cases

The current `src/model_runner.py` runs seven representative test cases:

1. Cell membrane summary
2. Plasma membrane flashcards
3. Phospholipid-bilayer multiple-choice question
4. Fluid mosaic model explanation
5. Membrane protein definition
6. Selective permeability explanation
7. Diffusion explanation

The current retrieval setting is:

```text
top_k = 1
```

---

## Generated Outputs

Generated files are stored under:

```text
outputs/
```

### `outputs/sample_outputs.json`

This file contains seven representative inference results.

Each result includes:

- Task type
- Student query
- Generation instruction
- Number of retrieved passages
- Retrieved passage ID
- Retrieved topic
- Source filename
- Retrieved passage text
- Retrieval similarity score
- Baseline output
- RAG output

### `outputs/human_evaluation_template.csv`

This file provides fields for human reviewers to score:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness
- Comments

### `outputs/README.md`

This file briefly explains what the generated output files contain and notes that the outputs are preliminary.

---

## Preliminary Results

The current pipeline successfully:

- Processed two biology source files
- Created 22 overlapping passages
- Generated embeddings with shape `(22, 384)`
- Saved passage metadata for all 22 passages
- Retrieved relevant passages using FAISS
- Loaded pretrained `google/flan-t5-base`
- Ran seven representative inference cases
- Generated baseline responses without retrieval
- Generated RAG responses using retrieved context
- Saved all seven results to `outputs/sample_outputs.json`
- Created a human-evaluation template
- Ran successfully from the repository root on a CPU
- Regenerated outputs using deterministic four-beam search

### Retrieval Results

The selective-permeability query retrieved:

```text
passive_transport_001
```

with a similarity score of approximately:

```text
0.7586
```

The diffusion query retrieved:

```text
passive_transport_018
```

with a similarity score of approximately:

```text
0.7486
```

These results confirm that the updated FAISS index retrieves relevant passages from the newly added passive-transport source.

### Initial Output Observations

The retrieval component is functioning correctly, and retrieved passages are generally relevant to the student queries.

Generation quality remains inconsistent despite the use of deterministic beam search.

Observed results include:

- The summary output included relevant membrane information but repeated a sentence
- The flashcard output did not create three correctly formatted flashcards
- The multiple-choice output did not include answer choices, a correct answer, or an explanation
- The fluid mosaic output was relevant but shorter than requested
- The membrane protein response included relevant source material but also included unnecessary carbohydrate details
- The selective-permeability output was relevant but incomplete
- The diffusion output did not fully answer the question even though the correct source passage was retrieved
- Baseline outputs were frequently unrelated or factually weak
- RAG outputs were generally more connected to the biology material than baseline outputs

These findings demonstrate a functioning end-to-end RAG pipeline while showing that prompt following, answer completeness, and structured output formatting require further improvement.

---

## Evaluation Plan

Both team members will independently review selected baseline and RAG outputs.

Each output will be scored from 1 to 5 for:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness

Multiple-choice questions will also be checked for:

- One clear correct answer
- Plausible incorrect options
- An explanation supported by the retrieved context
- Correct use of the requested format

Flashcards will be checked for:

- The requested number of flashcards
- Clear terms
- Accurate definitions
- Support from the retrieved source passage

Large reviewer disagreements will be discussed before final results are summarized.

The detailed evaluation rubric is available in:

```text
docs/evaluation_rubric.md
```

---

## Documentation

Additional project documentation is stored under:

```text
docs/
```

The documentation includes:

- `evaluation_rubric.md` — criteria for reviewing generated outputs
- `method_selection.md` — explanation of model and method selection
- `preliminary_results.md` — supporting notes about preliminary experiments

---

## Configuration

Model, retrieval, preprocessing, and path settings are documented in:

```text
configs/model_config.yaml
```

The current configuration file includes:

```yaml
generator:
  model_name: google/flan-t5-base
  max_input_length: 1024
  max_new_tokens: 250
  num_beams: 4
  do_sample: false
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3
  early_stopping: true

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
```

The YAML file currently serves as a centralized configuration reference.

The executable Python scripts still define and use their settings directly in code rather than loading all values from the YAML file at runtime.

The YAML values have been aligned with the current script settings so that the documented configuration accurately reflects the Milestone 4 pipeline.

Loading all settings dynamically from the configuration file remains a possible future improvement.

---

## Reproducibility Notes

The complete pipeline was tested using:

- Python 3.9.7
- macOS
- CPU execution
- PyTorch 2.2.2
- NumPy 1.26.4
- Transformers 4.57.6
- Sentence Transformers 5.1.2
- FAISS CPU 1.13.0

The workflow was tested inside a Python virtual environment.

The first execution downloads pretrained models from Hugging Face. Internet access is required during the initial model downloads.

FLAN-T5-base can run on a CPU, but generation is slower than it would be with a compatible GPU.

The embedding file is generated locally and must be rebuilt after cloning the repository.

Deterministic beam search is used instead of random sampling to improve consistency across repeated runs.

---

## Current Limitations

- The dataset currently contains only two introductory biology source files
- The current content is limited mainly to cell membranes and passive transport
- The final project requires additional source pages and biology topics
- Retrieval currently uses one passage per query
- Full comparisons of `k = 1`, `k = 3`, and `k = 5` remain planned
- FLAN-T5-base does not consistently follow requested output formats
- Some generated responses are incomplete
- Some generated responses are overly brief or factually weak
- Beam search improves reproducibility but does not guarantee complete or correctly formatted answers
- Baseline generation may produce unrelated or incorrect content
- Generated study materials require human review
- Lightweight personalization has not yet been fully implemented
- Current evaluation is preliminary
- CPU inference may be slow
- The embedding index must be rebuilt whenever the passage dataset changes
- The Python scripts do not yet load all settings directly from `model_config.yaml`
- The reusable helper module is not yet fully integrated into `model_runner.py`

---

## Troubleshooting

### The Embeddings Are Missing

Run:

```bash
python3 src/preprocessing.py
python3 src/embeddings.py
```

Then run:

```bash
python3 src/model_runner.py
```

### New Source Material Is Not Being Retrieved

Adding a raw text file does not automatically update the embedding index.

Run:

```bash
python3 src/preprocessing.py
python3 src/embeddings.py
python3 src/model_runner.py
```

### The Pipeline Saves Fewer Than Seven Results

Confirm that `src/model_runner.py` contains all seven test cases.

Then rerun:

```bash
python3 src/model_runner.py
```

The expected final message is:

```text
Saved 7 results to outputs/sample_outputs.json
```

### TensorFlow CPU Feature Message

A message similar to the following may appear:

```text
This TensorFlow binary is optimized to use available CPU instructions
```

This is informational and does not mean that the pipeline failed.

### `torch==2.8.0` Cannot Be Installed

The tested project environment uses:

```text
torch==2.2.2
```

Install the versions listed in:

```text
requirements.txt
```

### `TypeError` Involving `str | None`

The `str | None` type-hint syntax requires Python 3.10 or newer.

The project uses Python 3.9-compatible syntax where needed.

### Model Download Takes a Long Time

The first run downloads pretrained models from Hugging Face.

FLAN-T5-base is a relatively large model, and CPU execution may be slow. Later executions should use locally cached model files.

---

## References

Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv preprint arXiv:2210.11416.

Johnson, J., Douze, M., and Jégou, H. (2017). *Billion-scale similarity search with GPUs*. arXiv preprint arXiv:1702.08734.

Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33.

Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.
