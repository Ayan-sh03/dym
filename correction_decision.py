from typing import List, Optional, Tuple
from dataclasses import dataclass
from candidate_ranking import RankedCandidate

@dataclass
class CorrectionDecision:
    should_suggest: bool
    suggestion: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

class CorrectionDecisionMaker:
    def __init__(self):
        """Initialize the decision maker with configurable thresholds."""
        self.thresholds = {
            'min_score': 0.4,            # Minimum final score to consider suggestion
            'score_diff': 0.15,          # Required difference between top and original scores
            'high_confidence': 0.75,      # Score threshold for high confidence suggestions
            'medium_confidence': 0.6,     # Score threshold for medium confidence suggestions
            'context_boost': 0.1,         # Additional score boost for strong context relevance
            'max_edit_distance': 3,       # Maximum allowed edit distance
            'strong_context': 0.5,        # Threshold for considering context strongly relevant
            'weak_context': 0.3           # Threshold for considering context weakly relevant
        }

    def _analyze_candidate_strength(self, candidate: RankedCandidate) -> Tuple[float, str]:
        """
        Analyze candidate strength using multiple signals.
        Returns (confidence_boost, reason).
        """
        confidence_boost = 0.0
        reasons = []

        # Check edit distance
        if candidate.edit_distance == 1:
            confidence_boost += 0.1
            reasons.append("minor spelling difference")
        elif candidate.edit_distance > self.thresholds['max_edit_distance']:
            confidence_boost -= 0.2
            reasons.append("significant spelling difference")

        # Check context relevance
        if candidate.context_score > self.thresholds['strong_context']:
            confidence_boost += self.thresholds['context_boost']
            reasons.append("strong contextual match")
        elif candidate.context_score > self.thresholds['weak_context']:
            confidence_boost += self.thresholds['context_boost'] * 0.5
            reasons.append("moderate contextual match")

        # Check source reliability
        if candidate.source == 'lookup':
            confidence_boost += 0.1
            reasons.append("direct lookup match")
        elif candidate.phonetic_match:
            confidence_boost += 0.05
            reasons.append("phonetic similarity")

        return confidence_boost, ", ".join(reasons)

    def make_decision(
        self,
        original_query: str,
        candidates: List[RankedCandidate],
        context: Optional[List[str]] = None
    ) -> CorrectionDecision:
        """
        Decide whether to suggest a correction based on candidate rankings and other factors.
        """
        if not candidates:
            return CorrectionDecision(
                should_suggest=False,
                reason="No correction candidates available"
            )

        # Get top candidate
        top_candidate = candidates[0]

        # If it's an exact match, no suggestion needed
        if top_candidate.word.lower() == original_query.lower():
            return CorrectionDecision(
                should_suggest=False,
                reason="Query is already correct"
            )

        # Calculate base confidence from candidate's final score
        base_confidence = top_candidate.final_score

        # Get additional confidence boost and reasons
        confidence_boost, reason_details = self._analyze_candidate_strength(top_candidate)

        # Calculate final confidence
        final_confidence = base_confidence + confidence_boost

        # Decision making logic
        should_suggest = (
            final_confidence >= self.thresholds['min_score'] and
            top_candidate.edit_distance <= self.thresholds['max_edit_distance']
        )

        # Determine suggestion confidence level and add context-aware messaging
        if should_suggest:
            if final_confidence >= self.thresholds['high_confidence']:
                confidence_description = "high"
            elif final_confidence >= self.thresholds['medium_confidence']:
                confidence_description = "medium"
            else:
                confidence_description = "low"

            context_note = ""
            if top_candidate.context_score > self.thresholds['strong_context']:
                context_note = " (context strongly supports this)"
            elif top_candidate.context_score > self.thresholds['weak_context']:
                context_note = " (context moderately supports this)"

            reason = f"Suggested with {confidence_description} confidence{context_note} ({reason_details})"
        else:
            reason = f"Confidence too low for suggestion ({reason_details})"

        return CorrectionDecision(
            should_suggest=should_suggest,
            suggestion=top_candidate.word if should_suggest else None,
            confidence=final_confidence,
            reason=reason
        )

def format_suggestion(decision: CorrectionDecision, original_query: str) -> str:
    """Format the 'Did you mean' suggestion message."""
    if not decision.should_suggest or not decision.suggestion:
        return ""

    return f'Did you mean "{decision.suggestion}" instead of "{original_query}"?'


# Example usage
if __name__ == "__main__":
    from candidate_generation import CandidateGenerator
    from candidate_ranking import CandidateRanker

    # Initialize components
    generator = CandidateGenerator("seed_data.json")
    ranker = CandidateRanker("seed_data.json")
    decision_maker = CorrectionDecisionMaker()

    # Test cases with various scenarios
    test_cases = [
        # Common misspellings
        ("weathr", ["check", "the"]),           # Minor spelling error
        ("tecnology", ["latest"]),              # Minor error with strong context
        ("recipie", ["cooking", "best"]),       # Common misspelling

        # Borderline cases
        ("weatherr", ["forecast"]),             # Double letter error
        ("techenology", ["computer"]),          # Multiple errors

        # Cases that shouldn't trigger correction
        ("weather", ["sunny"]),                 # Correct spelling
        ("xzywtr", ["check"]),                 # Too different

        # Context-dependent cases
        ("finance", ["money", "bank"]),         # Correct with relevant context
        ("finanse", ["money", "bank"]),         # Error with relevant context
        ("finanse", ["play", "game"])          # Error with irrelevant context
    ]

    print("\nSpell Correction Analysis")
    print("=" * 100)
    print(f"{'Query':<15} {'Context':<20} {'Suggestion':<15} {'Confidence':<10} {'Should Suggest':<15} {'Reason'}")
    print("-" * 100)

    for query, context in test_cases:
        # Generate and rank candidates
        candidates = generator.generate_candidates(query)
        ranked_candidates = ranker.rank_candidates(candidates, None, context)

        # Make correction decision
        decision = decision_maker.make_decision(query, ranked_candidates, context)

        # Format output
        context_str = " ".join(context)
        suggestion = decision.suggestion if decision.should_suggest else "-"
        print(f"{query:<15} {context_str:<20} {suggestion:<15} {decision.confidence:.3f}     {str(decision.should_suggest):<15} {decision.reason}")

    print("=" * 100)