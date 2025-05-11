# Candidate Generation Module Structure

## Class: CandidateGenerator

### 1. Dependencies
```python
from typing import List, Dict, Set
import Levenshtein  # For edit distance calculation
import jellyfish    # For phonetic matching (Soundex, Metaphone)
from dataclasses import dataclass
```

### 2. Data Structures
```python
@dataclass
class Candidate:
    word: str
    original: str
    edit_distance: int
    phonetic_match: bool
    source: str  # 'lookup' | 'edit_distance' | 'phonetic'
    confidence: float
```

### 3. Main Methods

#### a. Constructor
```python
def __init__(self, seed_data_path: str):
    """
    Initialize with seed data and build lookup tables
    - Load keywords and misspellings
    - Build phonetic index
    - Initialize caches
    """
```

#### b. Lookup Generation
```python
def generate_lookup_candidates(self, token: str) -> List[Candidate]:
    """
    Generate candidates from direct lookup in seed data
    - Check common words
    - Check misspellings mapping
    - Check category-specific terms
    """
```

#### c. Edit Distance Generation
```python
def generate_edit_distance_candidates(self, token: str, max_distance: int = 2) -> List[Candidate]:
    """
    Generate candidates within specified Levenshtein distance
    - Check against common words
    - Filter by maximum edit distance
    - Calculate confidence based on distance
    """
```

#### d. Phonetic Generation
```python
def generate_phonetic_candidates(self, token: str) -> List[Candidate]:
    """
    Generate candidates based on phonetic similarity
    - Use Soundex algorithm
    - Use Metaphone algorithm
    - Combine results with confidence scores
    """
```

#### e. Main Generation Pipeline
```python
def generate_candidates(self, token: str) -> List[Candidate]:
    """
    Main pipeline to generate all candidates
    1. Try direct lookup first
    2. Generate edit distance candidates
    3. Generate phonetic candidates
    4. Combine and deduplicate results
    5. Sort by confidence
    """
```

### 4. Helper Methods

#### a. Confidence Calculation
```python
def calculate_confidence(self,
                        candidate: str,
                        original: str,
                        match_type: str) -> float:
    """
    Calculate confidence score based on:
    - Edit distance
    - Phonetic similarity
    - Source of match
    - Word frequency in seed data
    """
```

#### b. Deduplication
```python
def deduplicate_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
    """
    Remove duplicate candidates keeping highest confidence
    """
```

### 5. Caching
```python
class CandidateCache:
    """
    LRU cache for frequently requested tokens
    - Cache size limit
    - Expiration time
    - Thread-safe implementation
    """
```

## Usage Example
```python
# Initialize generator
generator = CandidateGenerator("seed_data.json")

# Process a token
candidates = generator.generate_candidates("weathr")

# Results example:
# [
#   Candidate(word="weather", original="weathr", edit_distance=1,
#             phonetic_match=True, source="edit_distance", confidence=0.9),
#   Candidate(word="whether", original="weathr", edit_distance=2,
#             phonetic_match=True, source="phonetic", confidence=0.7)
# ]
```

## Performance Considerations
1. Cached lookups for frequent tokens
2. Parallel processing for multiple tokens
3. Early termination for exact matches
4. Indexed phonetic matching
5. Optimized edit distance calculations

## Dependencies Required
```bash
pip install python-Levenshtein
pip install jellyfish