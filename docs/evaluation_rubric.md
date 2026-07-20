# Human Evaluation Rubric

Generated outputs will be reviewed by at least two team members.

Each criterion will be scored from 1 to 5.

| Criterion | Description |
|---|---|
| Factual grounding | Claims are supported by the retrieved textbook context |
| Relevance | The response directly addresses the study prompt |
| Readability | The response is clear and appropriate for an introductory biology student |
| Completeness | The response includes the important requested information |
| Usefulness | The output would help a student study or understand the topic |
| MCQ quality | The question has one clear correct answer and an explanation supported by the source text |

## Rating Scale

- 1: Very poor
- 2: Poor
- 3: Acceptable
- 4: Good
- 5: Excellent

## Review Procedure

Two team members will independently review selected baseline and RAG outputs.

The reviewers will compare:

- FLAN-T5 outputs generated without retrieved context
- RAG outputs generated with retrieved biology passages
- Retrieval settings using 1, 3, and 5 passages

Reviewers will record scores and comments. Large disagreements will be discussed before results are summarized.
