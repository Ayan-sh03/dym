# Context-Aware Spell Correction System

A Python-based spell correction system that uses context, user history, and domain knowledge to provide intelligent spelling suggestions.

## Features

### 1. Context-Aware Corrections
- Considers surrounding words for better accuracy
- Differentiates between similar words based on context (e.g., "weather" vs "whether")
- Supports domain-specific corrections (tech, health, business, etc.)

### 2. Intelligent Scoring
- Combines multiple signals for confidence scoring:
  - Edit distance similarity
  - Phonetic matching
  - Context relevance
  - User history
  - Domain knowledge

### 3. User Personalization
- Maintains user-specific correction history
- Learns from user interactions
- Adjusts confidence based on past corrections

### 4. Category-Specific Handling
- Specialized handling for different domains:
  - Technical terms
  - Medical terminology
  - Business vocabulary
  - Educational terms

## Performance

- Success Rate: 87.50% on test suite
- Average Processing Time: < 1ms per query
- Confidence Thresholds:
  - High Context: 0.75+
  - Medium Context: 0.50+
  - Low/Irrelevant Context: 0.35+

## Known Limitations

1. **Multiple Word Corrections**
   - Current implementation focuses on single-word corrections
   - Multiple word queries (e.g., "komputer teknology") are not supported
   - Future Enhancement: Add support for multi-word correction

2. **Context Confidence Variability**
   - Some valid corrections may fall below confidence thresholds
   - Example: "recepie" → "recipe" (0.657) below high threshold (0.75)
   - Trade-off between precision and recall

3. **Domain Limitations**
   - Limited to domains defined in seed data
   - May not perform optimally for specialized terminology
   - Requires domain-specific training data

## Implementation Details

### Components
1. **Data Preprocessing** (`data_preprocessing.py`)
   - Query normalization
   - Tokenization
   - User profile management

2. **Candidate Generation** (`candidate_generation.py`)
   - Edit distance based candidates
   - Phonetic matching
   - Lookup-based suggestions

3. **Candidate Ranking** (`candidate_ranking.py`)
   - Context-based scoring
   - User history integration
   - Confidence calculation

4. **Correction Decision** (`correction_decision.py`)
   - Threshold-based decision making
   - Context-aware suggestion formatting
   - Confidence level assignment

### Usage Example
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

# Get suggestion
suggestion = system.format_response(result)
```

## Test Coverage

- Basic misspellings with context
- Category-specific corrections
- User history impact
- Context disambiguation
- Cross-domain usage
- Edge cases and validation

## TODOs and Future Improvements

### High Priority
1. **Multi-word Support** 🔄
   - [ ] Implement token-based correction pipeline
   - [ ] Add phrase-level context analysis
   - [ ] Handle compound words and hyphenation
   - [ ] Support word splitting/joining corrections

2. **Performance Optimization** 🚀
   - [ ] Add caching for frequent corrections
   - [ ] Implement batch processing for multiple queries
   - [ ] Optimize n-gram model calculations
   - [ ] Add async processing support

### Medium Priority
3. **Dynamic Thresholds** 📊
   - [ ] Implement adaptive confidence thresholds
   - [ ] Add machine learning based threshold adjustment
   - [ ] Create domain-specific threshold profiles
   - [ ] Add feedback loop for threshold optimization

4. **Extended Domain Support** 🌐
   - [ ] Add medical terminology database
   - [ ] Integrate technical documentation corpus
   - [ ] Support custom domain configuration
   - [ ] Add domain detection



### Documentation
5. **Documentation Improvements** 📝
   - [ ] Add API documentation
   - [ ] Create usage examples
   - [ ] Add performance benchmarks
   - [ ] Create contribution guidelines

## Dependencies

See `requirements.txt` for full list of dependencies:
- python-Levenshtein
- jellyfish
- numpy

## Setup and Installation

See `setup.md` for detailed setup instructions.