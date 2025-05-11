from typing import List, Dict, Set
import Levenshtein
import jellyfish
from dataclasses import dataclass
import json
from collections import defaultdict
from threading import Lock
from time import time

@dataclass
class Candidate:
    word: str
    original: str
    edit_distance: int
    phonetic_match: bool
    source: str  # 'lookup' | 'edit_distance' | 'phonetic'
    confidence: float

class CandidateCache:
    def __init__(self, max_size: int = 1000, expiration: int = 3600):
        self.cache: Dict[str, tuple[List[Candidate], float]] = {}
        self.max_size = max_size
        self.expiration = expiration  # in seconds
        self.lock = Lock()

    def get(self, key: str) -> List[Candidate] | None:
        with self.lock:
            if key in self.cache:
                candidates, timestamp = self.cache[key]
                if time() - timestamp <= self.expiration:
                    return candidates
                else:
                    del self.cache[key]
            return None

    def put(self, key: str, candidates: List[Candidate]):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest = min(self.cache.items(), key=lambda x: x[1][1])
                del self.cache[oldest[0]]
            self.cache[key] = (candidates, time())

class CandidateGenerator:
    def __init__(self, seed_data_path: str):
        """Initialize with seed data and build lookup tables."""
        self.seed_data = self._load_seed_data(seed_data_path)
        self.common_words = set(self.seed_data["keywords"]["common_words"])
        self.misspellings = self._build_misspellings_index()
        self.categories = self._build_category_index()
        self.phonetic_index = self._build_phonetic_index()
        self.cache = CandidateCache()

    def _load_seed_data(self, path: str) -> dict:
        """Load seed data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _build_misspellings_index(self) -> Dict[str, str]:
        """Build index of misspellings to correct words."""
        index = {}
        for correct, misspelled in self.seed_data["keywords"]["misspellings"].items():
            for wrong in misspelled:
                index[wrong] = correct
        return index

    def _build_category_index(self) -> Set[str]:
        """Build set of all category-specific terms."""
        terms = set()
        for category in self.seed_data["keywords"]["categories"].values():
            terms.update(category)
        return terms

    def _build_phonetic_index(self) -> Dict[str, Set[str]]:
        """Build phonetic index for all words."""
        index = defaultdict(set)
        all_words = (self.common_words | self.categories |
                    set(self.misspellings.values()))

        for word in all_words:
            soundex = jellyfish.soundex(word)
            metaphone = jellyfish.metaphone(word)
            index[soundex].add(word)
            index[metaphone].add(word)
        return index

    def generate_lookup_candidates(self, token: str) -> List[Candidate]:
        """Generate candidates from direct lookup in seed data."""
        candidates = []

        # Check exact match in common words
        if token in self.common_words:
            candidates.append(Candidate(
                word=token,
                original=token,
                edit_distance=0,
                phonetic_match=True,
                source='lookup',
                confidence=1.0
            ))
            return candidates  # Early return for exact matches

        # Check misspellings
        if token in self.misspellings:
            correct = self.misspellings[token]
            candidates.append(Candidate(
                word=correct,
                original=token,
                edit_distance=Levenshtein.distance(token, correct),
                phonetic_match=True,
                source='lookup',
                confidence=0.95
            ))

        return candidates

    def generate_edit_distance_candidates(self, token: str, max_distance: int = 2) -> List[Candidate]:
        """Generate candidates within specified Levenshtein distance."""
        candidates = []
        all_words = self.common_words | self.categories

        for word in all_words:
            distance = Levenshtein.distance(token, word)
            if 0 < distance <= max_distance:
                confidence = 1.0 - (distance / (max_distance + 1))
                candidates.append(Candidate(
                    word=word,
                    original=token,
                    edit_distance=distance,
                    phonetic_match=False,
                    source='edit_distance',
                    confidence=confidence
                ))

        return candidates

    def generate_phonetic_candidates(self, token: str) -> List[Candidate]:
        """Generate candidates based on phonetic similarity."""
        candidates = []
        token_soundex = jellyfish.soundex(token)
        token_metaphone = jellyfish.metaphone(token)

        # Combine matches from both algorithms
        phonetic_matches = (self.phonetic_index.get(token_soundex, set()) |
                          self.phonetic_index.get(token_metaphone, set()))

        for word in phonetic_matches:
            if word != token:  # Skip exact matches
                distance = Levenshtein.distance(token, word)
                candidates.append(Candidate(
                    word=word,
                    original=token,
                    edit_distance=distance,
                    phonetic_match=True,
                    source='phonetic',
                    confidence=0.8 if distance <= 2 else 0.6
                ))

        return candidates

    def calculate_confidence(self, candidate: str, original: str, match_type: str) -> float:
        """Calculate confidence score based on multiple factors."""
        base_confidence = {
            'lookup': 0.95,
            'edit_distance': 0.8,
            'phonetic': 0.7
        }[match_type]

        # Adjust for edit distance
        distance = Levenshtein.distance(candidate, original)
        distance_penalty = distance * 0.1

        # Adjust for word frequency (if it's a common word)
        frequency_bonus = 0.1 if candidate in self.common_words else 0

        return min(1.0, max(0.0, base_confidence - distance_penalty + frequency_bonus))

    def deduplicate_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove duplicate candidates keeping highest confidence."""
        seen = {}
        for candidate in candidates:
            if candidate.word not in seen or seen[candidate.word].confidence < candidate.confidence:
                seen[candidate.word] = candidate
        return list(seen.values())

    def generate_candidates(self, token: str) -> List[Candidate]:
        """Main pipeline to generate all candidates."""
        # Check cache first
        cached = self.cache.get(token)
        if cached:
            return cached

        # Generate candidates from all sources
        candidates = []

        # 1. Try direct lookup first
        lookup_candidates = self.generate_lookup_candidates(token)
        if lookup_candidates and lookup_candidates[0].confidence == 1.0:
            # If we have an exact match, return immediately
            self.cache.put(token, lookup_candidates)
            return lookup_candidates

        candidates.extend(lookup_candidates)

        # 2. Generate edit distance candidates
        candidates.extend(self.generate_edit_distance_candidates(token))

        # 3. Generate phonetic candidates
        candidates.extend(self.generate_phonetic_candidates(token))

        # 4. Deduplicate and sort by confidence
        candidates = self.deduplicate_candidates(candidates)
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Cache results
        self.cache.put(token, candidates)

        return candidates

# Example usage
if __name__ == "__main__":
    generator = CandidateGenerator("seed_data.json")

    test_tokens = ["weathr", "tecnology", "recipie", "vacashun"]

    for token in test_tokens:
        print(f"\nProcessing token: {token}")
        candidates = generator.generate_candidates(token)
        print("Top candidates:")
        for candidate in candidates[:3]:  # Show top 3 candidates
            print(f"- {candidate}")