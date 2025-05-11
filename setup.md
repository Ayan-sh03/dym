# Spell Correction System Setup Guide

## System Requirements
- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation Steps

1. **Create and activate a virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv env
   .\env\Scripts\activate

   # On Unix/MacOS
   python3 -m venv env
   source env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare seed data**
   - Ensure `seed_data.json` is present in the root directory
   - The seed data file should contain:
     - Common words
     - Known misspellings
     - Category-specific terms
     - Common phrases

4. **Verify installation**
   ```bash
   python spell_correction_system.py
   ```
   This will run the test cases and verify that all components are working correctly.

## System Components

1. **Data Preprocessing (`data_preprocessing.py`)**
   - Handles query normalization
   - Manages user profiles
   - Tokenizes input

2. **Candidate Generation (`candidate_generation.py`)**
   - Generates correction candidates
   - Uses edit distance and phonetic matching
   - Maintains lookup tables

3. **Candidate Ranking (`candidate_ranking.py`)**
   - Ranks correction candidates
   - Incorporates context relevance
   - Handles personalization

4. **Correction Decision (`correction_decision.py`)**
   - Makes final correction decisions
   - Manages confidence thresholds
   - Formats suggestions

## Usage

```python
from spell_correction_system import SpellCorrectionSystem

# Initialize the system
system = SpellCorrectionSystem("seed_data.json")

# Process a query
result = system.process_query(
    query="tecnology",
    context=["latest", "software"],
    user_id="user1"
)

# Get formatted suggestion
suggestion = system.format_response(result)
```

## Customization

- Adjust confidence thresholds in `correction_decision.py`
- Modify context scoring weights in `candidate_ranking.py`
- Update seed data categories in `seed_data.json`

## Troubleshooting

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

2. **Seed Data Issues**
   - Verify `seed_data.json` format
   - Check file permissions

3. **Performance Issues**
   - Consider reducing max edit distance
   - Optimize n-gram model parameters
   - Adjust cache sizes

## Support

For issues and contributions:
1. Check existing documentation
2. Review test cases
3. Contact system administrators