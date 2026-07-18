# IE7374 Personalized Retrieval Augmented AI Study Companion for Technical Learning

## Project Overview

This project develops a personalized AI study companion using Retrieval-Augmented Generation (RAG) for technical learning. The application uses introductory biology textbook material to generate grounded educational resources including summaries, flashcards, multiple-choice questions, concise explanations, and personalized study recommendations.

The system combines information retrieval with a pretrained generative language model to ensure generated content is based on provided educational sources rather than relying only on the model's general knowledge.

---

## Objectives

The primary objectives of this project are:

- Develop an AI-powered study assistant for technical learning.
- Implement a Retrieval-Augmented Generation pipeline.
- Evaluate how retrieval improves factual grounding and relevance of generated content.
- Investigate how different retrieval settings affect output quality.
- Explore lightweight personalization based on student learning performance.

---

## System Architecture

The project follows a RAG-based architecture:

1. **Data Collection**
   - Collect introductory biology textbook content from Biology LibreTexts.
   - Use only sources explicitly licensed under CC BY 4.0.
   - Store source metadata and attribution information.

2. **Text Preprocessing**
   - Clean extracted text.
   - Remove unnecessary formatting artifacts.
   - Split documents into searchable overlapping passages.

3. **Embedding and Retrieval**
   - Convert passages into vector embeddings.
   - Store embeddings in a searchable vector database.
   - Retrieve relevant passages based on student questions.

4. **Text Generation**
   - Provide retrieved passages and user instructions to FLAN-T5.
   - Generate educational materials such as:
     - Summaries
     - Flashcards
     - Practice questions
     - Concept explanations

5. **Evaluation**
   - Compare outputs generated with and without retrieval.
   - Analyze retrieval settings and output quality.

---

## Model

### Generative Model

**google/flan-t5-base**

FLAN-T5 is an instruction-tuned encoder-decoder Transformer model designed to follow natural language instructions. It is used to generate structured educational content from retrieved textbook passages.

### Retrieval-Augmented Generation

The RAG pipeline improves grounding by retrieving relevant biology passages before generation. This reduces unsupported responses and encourages answers based on the selected educational materials.

---

## Dataset

The dataset consists of selected introductory biology pages from Biology LibreTexts.

Selected topics include:

- Cell structure
- Cell membranes
- Diffusion
- Cellular respiration
- Photosynthesis
- DNA
- Gene expression
- Genetics
- Evolution

Only pages with explicitly stated CC BY 4.0 licensing are included.

Each source is documented with:

- Page title
- Author/contributors
- Source URL
- License information

---

## Repository Structure
IE7374-RAG-AI-Study-Companion/

│
├── data/
│ └── attribution.csv
│
├── docs/
│ └── Project documentation and reports
│
├── experiments/
│ └── Experimental notebooks and results
│
├── models/
│ └── Model configurations
│
├── outputs/
│ └── Generated study materials
│
├── src/
│ ├── preprocessing.py
│ ├── embeddings.py
│ ├── retrieval.py
│ ├── generator.py
│ ├── evaluation.py
│ └── model_runner.py
│
├── requirements.txt
└── README.md
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jefler10/IE7374-RAG-AI-Study-Companion.git