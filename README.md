# IE7374 Personalized Retrieval-Augmented AI Study Companion for Technical Learning

## Project Overview

This project develops a personalized AI study companion using Retrieval-Augmented Generation (RAG) for technical learning. The application uses introductory biology textbook material to generate grounded educational resources, including summaries, flashcards, multiple-choice questions, concise explanations, and personalized study recommendations.

The system combines semantic information retrieval with a pretrained generative language model. Relevant textbook passages are retrieved before generation so that the educational content is based on selected source material rather than relying only on the model's general knowledge.

---

## Team Members

- Valika Chu
- Jeffrey Lara

---

## Objectives

The primary objectives of this project are to:

- Develop an AI-powered study assistant for technical learning.
- Implement an end-to-end Retrieval-Augmented Generation pipeline.
- Retrieve biology passages that are relevant to student questions.
- Generate summaries, flashcards, multiple-choice questions, and concept explanations.
- Evaluate how retrieval affects factual grounding and relevance.
- Investigate how different retrieval settings affect output quality.
- Explore lightweight personalization based on topic-level student performance.

---

## Research Questions

This project investigates the following questions:

1. How does RAG affect the factual grounding and relevance of FLAN-T5-generated study materials compared with generation without retrieved textbook passages?
2. How does the number of retrieved passages affect output quality when using `k = 1`, `k = 3`, and `k = 5`?
3. Can lightweight topic-level performance tracking improve the usefulness of practice questions and study recommendations?

---

## System Architecture

The project follows a RAG-based architecture:

1. **Data Collection**
   - Collect introductory biology material from Biology LibreTexts.
   - Use only pages whose individual license statements explicitly identify the text as CC BY 4.0.
   - Store source metadata and attribution information.

2. **Text Preprocessing**
   - Load raw text files.
   - Remove unnecessary whitespace and formatting artifacts.
   - Split documents into overlapping passages.
   - Save the passages in JSON Lines format.

3. **Embedding Generation**
   - Convert passages into numerical embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
   - Normalize the embeddings for cosine-similarity retrieval.
   - Save passage metadata for later retrieval.

4. **Semantic Retrieval**
   - Convert a student question into an embedding.
   - Search passage embeddings using FAISS.
   - Return the most relevant passages and similarity scores.

5. **Text Generation**
   - Load the pretrained `google/flan-t5-base` model.
   - Generate an instruction-only baseline output.
   - Generate a RAG output using the same instruction plus retrieved textbook context.
   - Save the generated results for comparison.

6. **Evaluation**
   - Compare baseline and RAG outputs.
   - Review factual grounding, relevance, readability, completeness, and usefulness.
   - Compare retrieval settings using one, three, and five passages.
   - Use at least two team members for selected human evaluations.

---

## Model and Framework Selection

### Generative Model

The primary generative model is:

```text
google/flan-t5-base
```

FLAN-T5 is a pretrained, instruction-tuned encoder-decoder Transformer model. It is designed to follow natural-language instructions, making it suitable for structured tasks such as generating flashcards, summaries, questions, and explanations.

FLAN-T5-base was selected because it offers stronger generation capacity than FLAN-T5-small while remaining practical for the project timeline and available computing resources.

### Embedding Model

The passage and query embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model produces compact semantic embeddings that can be used to compare the meaning of student questions with the meaning of textbook passages.

### Retrieval Library

The project uses FAISS for efficient vector-similarity search. Normalized passage and query embeddings are compared using inner-product search, which functions as cosine similarity when the vectors are normalized.

### Frameworks and Libraries

The main implementation uses:

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

This project does not train a language model from scratch and does not fine-tune FLAN-T5 during Milestone 3.

The system uses:

- pretrained `google/flan-t5-base` for text-generation inference;
- pretrained `sentence-transformers/all-MiniLM-L6-v2` for passage and query embeddings.

Because no supervised fine-tuning is performed, the project does not use conventional model-training, validation, and test splits. Instead, it uses a separate collection of evaluation prompts to test retrieval and generation behavior.

The project focuses on:

- source collection and attribution;
- text cleaning and passage creation;
- embedding generation;
- semantic retrieval;
- prompt construction;
- baseline and RAG generation;
- preliminary human evaluation.

---

## Dataset

The planned dataset consists of selected introductory biology pages from Biology LibreTexts.

Planned topics include:

- Cell structure
- Cell membranes
- Diffusion
- Cellular respiration
- Photosynthesis
- DNA
- Gene expression
- Genetics
- Evolution

Only pages with individual license statements explicitly identifying the text as CC BY 4.0 are included. Pages using CK-12, CC BY-NC, undeclared, or other licenses are excluded.

Each approved source is documented with:

- Page identifier
- Topic
- Page title
- Author or contributors
- Source URL
- License
- Access date

For the Milestone 3 feasibility implementation, the repository includes a cleaned cell-membrane text sample from a CC BY 4.0 Biology LibreTexts page. The source collection will be expanded in later project stages.

The initial source text is split into overlapping passages of approximately 250 words with a 50-word overlap. The final pipeline may use tokenizer-based chunks of approximately 300–400 tokens as the dataset expands.

---

## Repository Structure

```text
IE7374-RAG-AI-Study-Companion/
│
├── data/
│   ├── raw/
│   │   └── cell_membrane.txt
│   ├── processed/
│   │   ├── passages.jsonl
│   │   └── passage_metadata.json
│   ├── attribution.csv
│   └── evaluation_prompts.json
│
├── docs/
│   ├── README.md
│   ├── method_selection.md
│   ├── evaluation_rubric.md
│   └── preliminary_results.md
│
├── models/
│   └── config.yaml
│
├── outputs/
│   ├── sample_outputs.json
│   └── human_evaluation_template.csv
│
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── evaluation.py
│   └── model_runner.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The generated embedding file is not committed because it can be recreated locally:

```text
data/processed/embeddings.npy
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jefler10/IE7374-RAG-AI-Study-Companion.git
cd IE7374-RAG-AI-Study-Companion
```


### 2. Create a virtual environment

The project was tested with Python 3.9.7.

```bash
python3 -m venv venv
```

Activate the environment on macOS or Linux:

```bash
source venv/bin/activate
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

After activation, the command-line prompt should begin with:

```text
(venv)
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

The initial installation may take several minutes because PyTorch, Transformers, Sentence Transformers, and FAISS must be installed.

---

## Running the Pipeline

Run all commands from the repository root.

### 1. Preprocess the raw biology text

```bash
python3 src/preprocessing.py
```

Expected output:

```text
Saved 3 passages to data/processed/passages.jsonl
```

### 2. Create passage embeddings

```bash
python3 src/embeddings.py
```

Expected output:

```text
Saved embeddings with shape (3, 384)
Saved passage metadata to data/processed/passage_metadata.json
```

The first run downloads the sentence-transformer model.

### 3. Test semantic retrieval

```bash
python3 src/retrieval.py
```

The script runs this sample query:

```text
What is the purpose of the cell membrane?
```

It displays the three highest-ranked passages and their similarity scores.

### 4. Test FLAN-T5 generation

```bash
python3 src/generator.py
```

The first run downloads `google/flan-t5-base`, which is approximately 1 GB. Generation may take longer on a CPU.

### 5. Run the complete baseline and RAG pipeline

```bash
python3 src/model_runner.py
```

This script:

- loads the passage retriever;
- loads FLAN-T5-base;
- runs four educational tasks;
- generates an instruction-only baseline;
- retrieves three relevant passages;
- generates a RAG output;
- saves the results.

Expected output:

```text
Saved 4 results to outputs/sample_outputs.json
```

### 6. Create the human-evaluation template

```bash
python3 src/evaluation.py
```

Expected output:

```text
Saved evaluation template to outputs/human_evaluation_template.csv
```

---

## Expected Outputs

The pipeline creates or updates the following files:

```text
data/processed/passages.jsonl
data/processed/embeddings.npy
data/processed/passage_metadata.json
outputs/sample_outputs.json
outputs/human_evaluation_template.csv
```

`outputs/sample_outputs.json` contains:

- task type;
- student query;
- generation instruction;
- number of retrieved passages;
- retrieved passage text;
- retrieval scores;
- baseline output;
- RAG output.

`outputs/human_evaluation_template.csv` contains fields for reviewers to score:

- factual grounding;
- relevance;
- readability;
- completeness;
- usefulness;
- comments.

---

## Preliminary Results

The initial Milestone 3 pipeline successfully:

- processed one CC BY 4.0 biology source;
- converted the source into three overlapping passages;
- generated embeddings with shape `(3, 384)`;
- retrieved relevant cell-membrane passages using FAISS;
- loaded `google/flan-t5-base`;
- generated biology text using instructions and retrieved context;
- created four baseline and RAG comparisons;
- saved generated results to `outputs/sample_outputs.json`;
- created a human-evaluation CSV template;
- ran successfully in a fresh Python virtual environment.

For the query:

```text
What is the purpose of the cell membrane?
```

the top three similarity scores were:

```text
0.5188
0.5038
0.4973
```

All three retrieved passages were related to plasma-membrane structure, transport, or function.

Prompt refinement and repetition controls were added after preliminary FLAN-T5 outputs repeated similar phrases. The current generation configuration uses beam search, a repetition penalty, and an n-gram repetition restriction.

Additional experiments will expand the dataset and compare:

```text
k = 1
k = 3
k = 5
```

---

## Evaluation Plan

At least one team member will independently review selected baseline and RAG outputs.

Each output will be scored from 1 to 5 for:

- Factual grounding
- Relevance
- Readability
- Completeness
- Usefulness

Multiple-choice questions will also be checked for:

- one clear correct answer;
- plausible incorrect options;
- an explanation supported by the source context.

Large reviewer disagreements will be discussed before the results are summarized.

The detailed evaluation rubric is available in:

```text
docs/evaluation_rubric.md
```

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

The full workflow was successfully tested inside a newly created virtual environment.

The first execution downloads pretrained models from Hugging Face. Internet access is therefore required during the initial run.

FLAN-T5-base can run on CPU, but generation is slower than it would be with a compatible GPU.

---

## Current Limitations

- The Milestone 3 feasibility dataset currently contains only one source page.
- The final project requires a larger collection covering additional biology topics.
- Retrieval rankings are based on a very small passage collection.
- FLAN-T5-base may not always follow the requested output format exactly.
- Generated materials still require human review.
- Lightweight personalization has not yet been fully implemented.
- The current experiments primarily use `k = 3`; full comparisons of `k = 1`, `k = 3`, and `k = 5` remain planned.

---

## Troubleshooting

### `torch==2.8.0` cannot be installed

The project uses:

```text
torch==2.2.2
```

because it is compatible with the tested Python 3.9.7 environment.

### `TypeError` involving `str | None`

The `str | None` type-hint syntax requires Python 3.10 or newer. The project uses `context=None` for compatibility with Python 3.9.

### Embeddings are missing

Run:

```bash
python3 src/preprocessing.py
python3 src/embeddings.py
```

before running retrieval or the complete pipeline.

---

## References

Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv preprint arXiv:2210.11416.

Johnson, J., Douze, M., and Jégou, H. (2017). *Billion-scale similarity search with GPUs*. arXiv preprint arXiv:1702.08734.

Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33.

Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.
