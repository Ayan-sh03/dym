from typing import Optional, List, Dict
from dataclasses import dataclass
import json

from data_preprocessing import DataPreprocessor
from candidate_generation import CandidateGenerator
from candidate_ranking import CandidateRanker
from correction_decision import CorrectionDecisionMaker, format_suggestion

@dataclass
class SpellCorrectionResult:
    original_query: str
    correction: Optional[str]
    should_suggest: bool
    confidence: float
    reason: str
    context_used: List[str]

class SpellCorrectionSystem:
    def __init__(self, seed_data_path: str):
        """Initialize the spell correction system with all components."""
        self.preprocessor = DataPreprocessor(seed_data_path)
        self.generator = CandidateGenerator(seed_data_path)
        self.ranker = CandidateRanker(seed_data_path)
        self.decision_maker = CorrectionDecisionMaker()

        # Load user profiles if any exist
        self.user_profiles: Dict[str, List[str]] = {}

    def process_query(
        self,
        query: str,
        context: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> SpellCorrectionResult:
        """
        Process a query through the complete spell correction pipeline.

        Args:
            query: The user's search query
            context: Optional list of context words
            user_id: Optional user ID for personalization

        Returns:
            SpellCorrectionResult containing correction details
        """
        # 1. Preprocess the query
        processed_query, tokens, user_profile = self.preprocessor.process_query(query, user_id)

        # 2. Generate correction candidates
        candidates = self.generator.generate_candidates(processed_query)

        # 3. Rank candidates using context and user profile
        ranked_candidates = self.ranker.rank_candidates(candidates, user_id, context)

        # 4. Make correction decision
        decision = self.decision_maker.make_decision(processed_query, ranked_candidates, context)

        # 5. Update user profile if correction is accepted
        if decision.should_suggest and user_id:
            self.ranker.update_user_profile(user_id, decision.suggestion or processed_query)

        # 6. Return result
        return SpellCorrectionResult(
            original_query=query,
            correction=decision.suggestion,
            should_suggest=decision.should_suggest,
            confidence=decision.confidence,
            reason=decision.reason,
            context_used=context or []
        )

    def format_response(self, result: SpellCorrectionResult) -> str:
        """Format the correction result for display."""
        if not result.should_suggest:
            return ""

        return format_suggestion(
            result=result,
            original_query=result.original_query
        )


# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = SpellCorrectionSystem("seed_data.json")

    # Test cases that demonstrate different aspects of the system
    test_cases = [
        # High confidence corrections with strong context
        ("tecnology", ["latest", "software"], "user1"),
        ("recipie", ["cooking", "best"], "user2"),

        # Medium confidence with moderate context
        ("weathr", ["check", "the"], "user3"),

        # Corrections with user history
        ("tecnology", ["computer"], "user1"),  # Same user, different context

        # Cases that shouldn't trigger correction
        ("weather", ["sunny", "day"], "user4"),
        ("xzywtr", ["what", "is"], "user5"),

        # Context-dependent corrections
        ("finanse", ["money", "bank"], "user6"),
        ("finanse", ["play", "game"], "user6")  # Same misspelling, different context
    ]

    # Process each test case
    print("\nSpell Correction System Analysis")
    print("=" * 100)
    print(f"{'Query':<15} {'Context':<25} {'User':<10} {'Correction':<15} {'Confidence':<10} {'Suggest?':<8}")
    print("-" * 100)

    for query, context, user_id in test_cases:
        result = system.process_query(query, context, user_id)

        # Format output
        context_str = " ".join(context)
        correction = result.correction if result.should_suggest else "-"
        confidence = f"{result.confidence:.3f}" if result.should_suggest else "-"

        print(f"{query:<15} {context_str:<25} {user_id:<10} {correction:<15} {confidence:<10} {str(result.should_suggest):<8}")

        if result.should_suggest:
            print(f"Reason: {result.reason}")
            print(f"Suggestion: {system.format_response(result)}")
        print("-" * 100)