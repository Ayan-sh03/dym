# Spell Correction System ("Did You Mean")

This project implements a modular spell correction system designed to provide "Did you mean" suggestions for user queries. It leverages techniques like data preprocessing, candidate generation (including indexed lookup, edit distance, and phonetic matching), candidate ranking (using n-gram probabilities and personalization), and a correction decision module.

## Project Overview

The system takes a user's search query and, if a likely misspelling is detected, suggests a correction. The core logic is encapsulated in the `SpellCorrectionSystem` class, which orchestrates the following components:

-   **`DataPreprocessor`**: Normalizes and tokenizes input queries.
-   **`CandidateGenerator`**: Generates potential correction candidates.
-   **`CandidateRanker`**: Ranks the generated candidates based on relevance and user history.
-   **`CorrectionDecisionMaker`**: Decides if a correction should be suggested.

## Key Features

-   Modular architecture for easy extension and maintenance.
-   Utilizes edit distance (Levenshtein) and phonetic matching for candidate generation.
-   Incorporates n-gram language models and personalization for ranking.
-   Provides confidence scores for suggestions.

## Requirements

-   Python 3.9+
-   `numpy`
-   `python-Levenshtein`
-   `jellyfish`
-   `typing-extensions`

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

The main script `spell_correction_system.py` includes an example usage section. You can run it directly to see the system in action with predefined test cases:

```bash
python spell_correction_system.py
```

This will output an analysis of various queries, showing the original query, context, suggested correction (if any), confidence, and whether a suggestion was made.

## Project Structure

-   `spell_correction_system.py`: Main class orchestrating the spell correction pipeline.
-   `data_preprocessing.py`: Handles query normalization and tokenization.
-   `candidate_generation.py`: Generates correction candidates.
-   `candidate_ranking.py`: Ranks candidates.
-   `correction_decision.py`: Decides whether to offer a suggestion.
-   `seed_data.json`: Contains initial keywords for the system.
-   `plan.md`: Detailed project plan and architecture.
-   `requirements.txt`: Project dependencies.
-   `test_spell_correction.py`: (Assumed, based on file list) Unit tests for the system.

## Future Enhancements (from plan.md)

-   Dynamic model retraining.
-   Enhanced personalization.
-   Continuous performance optimization.