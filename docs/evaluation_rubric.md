# Human Evaluation Rubric

Generated outputs were independently reviewed by two evaluators.

Each response was scored from 1 to 5 on the following criteria:

| Criterion | Description |
|---|---|
| Factual grounding | Claims are supported by the retrieved biology source material |
| Relevance | The response directly addresses the study prompt |
| Readability | The response is clear and appropriate for an introductory biology student |
| Completeness | The response includes the important requested information |
| Usefulness | The response would help a student study or understand the topic |

## Rating Scale

- 1: Very poor
- 2: Poor
- 3: Acceptable
- 4: Good
- 5: Excellent

## Review Procedure

Two evaluators independently reviewed all 28 generated responses across four experimental conditions:

- Instruction-only baseline
- RAG with `k = 1`
- RAG with `k = 3`
- RAG with `k = 5`

Each evaluator recorded scores for factual grounding, relevance, readability, completeness, and usefulness, along with optional comments.

The completed evaluation files are stored in:

    outputs/human_evaluation_reviewer1.csv
    outputs/human_evaluation_reviewer2.csv
    outputs/human_evaluation_combined_detailed.csv
    outputs/human_evaluation_combined_summary.csv

The two evaluator scores were combined and averaged to produce the final results reported in the main README.
