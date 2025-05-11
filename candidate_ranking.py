from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import numpy as np
from collections import defaultdict
from candidate_generation import Candidate
import math
from threading import Lock
import time

@dataclass
class RankedCandidate(Candidate):
    global_score: float
    personal_score: float
    final_score: float
    context_score: float

class NGramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.total_ngrams = 0
        self.vocabulary = set()

    def train(self, sentences: List[str]):
        """Train the n-gram model on a list of sentences."""
        for sentence in sentences:
            tokens = sentence.split()
            self.vocabulary.update(tokens)

            # Add start and end tokens
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            # Count n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.ngram_counts[ngram] += 1
                self.total_ngrams += 1

    def score(self, sentence: str) -> float:
        """Calculate the average log probability of a sentence."""
        tokens = sentence.split()
        if not tokens:
            return -float('inf') # Or a very large negative number

        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

        log_prob = 0.0
        num_ngrams_processed = 0

        # Ensure vocabulary size is not zero for smoothing
        vocab_size = len(self.vocabulary)
        denominator = self.total_ngrams + vocab_size if vocab_size > 0 else self.total_ngrams + 1


        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            count = self.ngram_counts[ngram]
            # Add-1 smoothing
            prob = (count + 1) / denominator
            if prob > 0: # Avoid math.log(0)
                 log_prob += math.log(prob)
            else: # Handle zero probability with a very small log probability
                log_prob += -float('inf')
            num_ngrams_processed += 1

        if num_ngrams_processed == 0:
             # This case should ideally not happen if tokens is not empty
             # and n >= 1, but as a fallback.
            return -float('inf')

        return log_prob / num_ngrams_processed

class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.query_history: Dict[str, int] = defaultdict(int)
        self.category_preferences: Dict[str, float] = defaultdict(float)
        self.last_update = time.time()

    def update(self, query: str, category: Optional[str] = None):
        """Update user profile with new query."""
        self.query_history[query] += 1
        if category:
            self.category_preferences[category] += 1
        self.last_update = time.time()

class CandidateRanker:
    def __init__(self, seed_data_path: str):
        """Initialize the ranker with seed data and models."""
        self.seed_data = self._load_seed_data(seed_data_path)
        self.ngram_model = self._initialize_language_model()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.profile_lock = Lock()

        # Configuration weights
        self.weights = {
            'global': 0.4,    # Global language model score
            'personal': 0.3,  # Personal history score
            'context': 0.3    # Context relevance score
        }

    def _load_seed_data(self, path: str) -> dict:
        """Load seed data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _initialize_language_model(self) -> NGramModel:
        """Initialize and train the n-gram language model."""
        model = NGramModel(n=2)

        training_data = []

        # Add phrases
        training_data.extend(self.seed_data["keywords"]["phrases"])

        # Add common words as single-word sentences
        training_data.extend(self.seed_data["keywords"]["common_words"])

        # Add category words as single-word sentences and joined category phrases
        all_category_words = []
        for category_words in self.seed_data["keywords"]["categories"].values():
            training_data.append(" ".join(category_words))
            all_category_words.extend(category_words)
        training_data.extend(all_category_words)

        # Create some simple two-word phrases from common words for bi-gram model
        if model.n >= 2:
            common_words_list = list(self.seed_data["keywords"]["common_words"])
            if len(common_words_list) >= 2:
                for i in range(len(common_words_list) -1):
                    # Create pairs like "word1 word2"
                    training_data.append(f"{common_words_list[i]} {common_words_list[i+1]}")

            # Create pairs from category words
            if len(all_category_words) >=2:
                 for i in range(len(all_category_words) -1):
                    if all_category_words[i] != all_category_words[i+1]: # Avoid "word word"
                        training_data.append(f"{all_category_words[i]} {all_category_words[i+1]}")

        # Remove duplicates that might have been added
        training_data = list(set(training_data))

        model.train(training_data)
        return model

    def _calculate_global_score(self, candidate: Candidate) -> float:
        """Calculate global score using language model."""
        # Combine confidence from candidate generator with language model score
        lm_score = self.ngram_model.score(candidate.word)

        # Convert log probability to a raw probability (0 to 1)
        # Add a small constant to lm_score to avoid math.exp underflow for very negative lm_score
        # and to ensure that even very unlikely sequences get a tiny non-zero probability.
        # However, for an average log probability, this might not be as extreme.
        # Let's cap the lm_score to avoid exp overflow if it's positive (though unlikely for log probs)
        # and underflow if it's too negative.
        capped_lm_score = max(-700, min(lm_score, 0)) # Typical log probs are negative
        normalized_lm_score = math.exp(capped_lm_score)


        # Weight between original confidence and language model score
        return 0.7 * candidate.confidence + 0.3 * normalized_lm_score

    def _calculate_personal_score(self, candidate: Candidate, user_id: Optional[str]) -> float:
        """Calculate personalization score based on user history."""
        if not user_id or user_id not in self.user_profiles:
            return 0.5  # Neutral score if no user profile

        profile = self.user_profiles[user_id]

        # Check word frequency in user's history
        word_freq = profile.query_history.get(candidate.word, 0)
        max_freq = max(profile.query_history.values()) if profile.query_history else 1

        # Normalize frequency score
        freq_score = word_freq / max_freq if max_freq > 0 else 0

        # Consider category preferences if word belongs to a category
        category_score = 0
        for category, words in self.seed_data["keywords"]["categories"].items():
            if candidate.word in words:
                category_score = profile.category_preferences.get(category, 0)
                break

        # Combine scores
        personal_score = 0.7 * freq_score + 0.3 * category_score
        return personal_score

    def _calculate_context_score(self, candidate: Candidate, context: Optional[List[str]] = None) -> float:
        """
        Calculate contextual relevance score using multiple signals:
        1. N-gram probability of the sequence
        2. Word co-occurrence in training data
        3. Position-based relevance
        """
        if not context:
            return 0.5  # Neutral score if no context

        # 1. N-gram sequence score
        context_str = " ".join(context + [candidate.word])
        ngram_score = self.ngram_model.score(context_str)
        capped_ngram_score = max(-700, min(ngram_score, 0))
        sequence_score = math.exp(capped_ngram_score)

        # 2. Word co-occurrence score
        cooccurrence_score = 0.0
        candidate_bigrams = set()
        # Get all bigrams from training data that contain the candidate word
        for ngram, count in self.ngram_model.ngram_counts.items():
            if candidate.word in ngram:
                candidate_bigrams.add(ngram)

        # Check if any context words appear in bigrams with the candidate
        matches = 0
        for context_word in context:
            for bigram in candidate_bigrams:
                if context_word in bigram:
                    matches += 1
                    break

        if context:  # Avoid division by zero
            cooccurrence_score = matches / len(context)

        # 3. Position-based score
        # Words closer to the candidate should have more impact
        position_scores = []
        for i, context_word in enumerate(context):
            position_weight = 1.0 / (len(context) - i)  # Later words get higher weights

            # Check if this word appears in any bigrams with the candidate
            word_score = 0.0
            for bigram in candidate_bigrams:
                if context_word in bigram:
                    word_score = position_weight
                    break
            position_scores.append(word_score)

        position_score = sum(position_scores) / len(context) if context else 0.0

        # Combine all scores with weights
        # Sequence probability is important but shouldn't dominate
        final_score = (
            0.4 * sequence_score +
            0.3 * cooccurrence_score +
            0.3 * position_score
        )

        return final_score

    def update_user_profile(self, user_id: str, query: str, category: Optional[str] = None):
        """Update user profile with new query information."""
        with self.profile_lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id)
            self.user_profiles[user_id].update(query, category)

    def rank_candidates(self,
                       candidates: List[Candidate],
                       user_id: Optional[str] = None,
                       context: Optional[List[str]] = None) -> List[RankedCandidate]:
        """Rank candidates using all scoring components."""
        ranked_candidates = []

        for candidate in candidates:
            # Calculate individual scores
            global_score = self._calculate_global_score(candidate)
            personal_score = self._calculate_personal_score(candidate, user_id)
            context_score = self._calculate_context_score(candidate, context)

            # Calculate final weighted score
            final_score = (
                self.weights['global'] * global_score +
                self.weights['personal'] * personal_score +
                self.weights['context'] * context_score
            )

            # Create RankedCandidate
            ranked_candidate = RankedCandidate(
                word=candidate.word,
                original=candidate.original,
                edit_distance=candidate.edit_distance,
                phonetic_match=candidate.phonetic_match,
                source=candidate.source,
                confidence=candidate.confidence,
                global_score=global_score,
                personal_score=personal_score,
                context_score=context_score,
                final_score=final_score
            )

            ranked_candidates.append(ranked_candidate)

        # Sort by final score
        ranked_candidates.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_candidates

# Example usage
if __name__ == "__main__":
    from candidate_generation import CandidateGenerator

    # Initialize components
    generator = CandidateGenerator("seed_data.json")
    ranker = CandidateRanker("seed_data.json")

    # Test queries with various contexts
    test_cases = [
        # Weather related contexts
        ("weathr", "user1", ["check", "the"]),
        ("weathr", "user2", ["forecast", "today"]),
        ("weathr", "user3", ["sunny", "cloudy"]),

        # Technology related contexts
        ("tecnology", "user1", ["latest"]),
        ("tecnology", "user2", ["computer", "software"]),
        ("tecnology", "user3", ["modern", "digital"]),

        # Recipe related contexts
        ("recipie", "user1", ["cooking", "best"]),
        ("recipie", "user2", ["food", "ingredients"]),
        ("recipie", "user3", ["traditional", "homemade"]),

        # Shopping related contexts
        ("shoping", "user1", ["online", "store"]),
        ("shoping", "user2", ["buy", "items"]),

        # Finance related contexts
        ("finanse", "user1", ["money", "bank"]),
        ("finanse", "user2", ["investment", "market"]),

        # Test with irrelevant contexts
        ("weathr", "user4", ["piano", "music"]),      # Unrelated context
        ("tecnology", "user4", ["garden", "plants"]),  # Unrelated context
        ("recipie", "user4", ["car", "drive"])        # Unrelated context
    ]

    # Print header for better formatted output
    print("\nContext Score Analysis")
    print("=" * 80)
    print(f"{'Query':<12} {'Correction':<12} {'Context':<30} {'Context Score':<8} {'Final Score':<8}")
    print("-" * 80)

    for query, user_id, context in test_cases:
        # Generate candidates
        candidates = generator.generate_candidates(query)

        # Rank candidates
        ranked_candidates = ranker.rank_candidates(candidates, user_id, context)

        if ranked_candidates:
            top_candidate = ranked_candidates[0]
            context_str = " ".join(context)
            print(f"{query:<12} {top_candidate.word:<12} {context_str:<30} {top_candidate.context_score:8.3f} {top_candidate.final_score:8.3f}")

            # Update user profile with selected correction
            ranker.update_user_profile(user_id, top_candidate.word)

    print("=" * 80)