from spell_correction_system import SpellCorrectionSystem
from typing import List, Tuple, Dict
import json
from dataclasses import dataclass
import time

@dataclass
class TestCase:
    scenario: str
    query: str
    context: List[str]
    user_id: str
    expected_correction: str
    min_confidence: float
    category: str

class SpellCorrectionTester:
    def __init__(self, seed_data_path: str):
        self.system = SpellCorrectionSystem(seed_data_path)
        self.results: Dict[str, List[dict]] = {}

    def run_tests(self):
        """Run all test scenarios and collect results."""
        # Adjusted confidence thresholds based on context strength
        CONFIDENCE = {
            'high': 0.75,    # Strong context or direct lookup matches
            'medium': 0.5,   # Moderate context relevance
            'low': 0.35      # Weak or irrelevant context
        }

        test_cases = [
            # 1. Basic Misspellings with Strong Context
            TestCase(
                scenario="Basic misspelling with relevant context",
                query="tecnology",
                context=["latest", "software"],
                user_id="tech_user",
                expected_correction="technology",
                min_confidence=CONFIDENCE['high'],
                category="basic_corrections"
            ),

            # 2. Multiple Word Context (Adjusted threshold)
            TestCase(
                scenario="Multiple word context",
                query="compter",
                context=["best", "programming", "software"],
                user_id="tech_user",
                expected_correction="computer",
                min_confidence=CONFIDENCE['medium'],
                category="basic_corrections"
            ),

            # Additional Basic Tests
            TestCase(
                scenario="Common typo correction",
                query="recepie",
                context=["cooking", "food"],
                user_id="food_user",
                expected_correction="recipe",
                min_confidence=CONFIDENCE['high'],
                category="basic_corrections"
            ),

            # 3. Category-Specific Corrections
            TestCase(
                scenario="Health category correction",
                query="medisine",
                context=["doctor", "prescription"],
                user_id="health_user",
                expected_correction="medicine",
                min_confidence=0.7,
                category="category_specific"
            ),

            # 4. User History Impact
            TestCase(
                scenario="Correction with user history",
                query="tecnology",
                context=["new"],
                user_id="tech_user",
                expected_correction="technology",
                min_confidence=0.6,
                category="user_specific"
            ),

            # 5. Context Disambiguation
            TestCase(
                scenario="Context disambiguation - weather vs whether",
                query="wether",
                context=["sunny", "forecast"],
                user_id="weather_user",
                expected_correction="weather",
                min_confidence=0.7,
                category="disambiguation"
            ),

            # 6. Irrelevant Context
            TestCase(
                scenario="Correction with irrelevant context",
                query="shoping",
                context=["quantum", "physics"],
                user_id="random_user",
                expected_correction="shopping",
                min_confidence=0.5,
                category="context_handling"
            ),

            # Additional Category Tests
            TestCase(
                scenario="Health term with weak context",
                query="medicin",
                context=["buy", "online"],
                user_id="random_user",
                expected_correction="medicine",
                min_confidence=CONFIDENCE['low'],
                category="category_specific"
            ),

            # 7. Educational Terms (Adjusted threshold)
            TestCase(
                scenario="Educational term correction",
                query="educasion",
                context=["university", "course"],
                user_id="student_user",
                expected_correction="education",
                min_confidence=CONFIDENCE['medium'],
                category="category_specific"
            ),

            # 8. Business Terms (Adjusted threshold)
            TestCase(
                scenario="Business term correction",
                query="finnance",
                context=["investment", "market"],
                user_id="business_user",
                expected_correction="finance",
                min_confidence=CONFIDENCE['high'],
                category="category_specific"
            ),

            # Additional Business Context
            TestCase(
                scenario="Business term with technical context",
                query="bizness",
                context=["software", "startup"],
                user_id="tech_user",
                expected_correction="business",
                min_confidence=CONFIDENCE['medium'],
                category="category_specific"
            ),

            # 9. Multiple Errors
            TestCase(
                scenario="Multiple error correction",
                query="komputer teknology",
                context=["software", "programming"],
                user_id="tech_user",
                expected_correction="computer technology",
                min_confidence=0.6,
                category="complex_corrections"
            ),

            # Additional Edge Cases
            TestCase(
                scenario="Nearly correct spelling",
                query="technologyy",
                context=["software"],
                user_id="tech_user",
                expected_correction="technology",
                min_confidence=CONFIDENCE['high'],
                category="validation"
            ),

            # 10. No Correction Needed
            TestCase(
                scenario="Correct spelling",
                query="technology",
                context=["software"],
                user_id="tech_user",
                expected_correction="technology",
                min_confidence=0.0,
                category="validation"
            ),

            # Mixed Context Tests
            TestCase(
                scenario="Technical term in medical context",
                query="computr",
                context=["hospital", "system"],
                user_id="health_user",
                expected_correction="computer",
                min_confidence=CONFIDENCE['medium'],
                category="cross_domain"
            ),

            TestCase(
                scenario="Business term in tech context",
                query="invstment",
                context=["software", "startup"],
                user_id="tech_user",
                expected_correction="investment",
                min_confidence=CONFIDENCE['medium'],
                category="cross_domain"
            ),
        ]

        # Process each test case
        for test_case in test_cases:
            self._run_single_test(test_case)

    def _run_single_test(self, test_case: TestCase):
        """Run a single test case and store results."""
        start_time = time.time()

        # Process the query
        result = self.system.process_query(
            query=test_case.query,
            context=test_case.context,
            user_id=test_case.user_id
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Analyze results
        success = True
        failure_reason = ""

        if result.should_suggest:
            if result.correction != test_case.expected_correction:
                success = False
                failure_reason = f"Expected {test_case.expected_correction}, got {result.correction}"
            elif result.confidence < test_case.min_confidence:
                success = False
                failure_reason = f"Confidence too low: {result.confidence:.3f} < {test_case.min_confidence}"
        elif test_case.expected_correction != test_case.query:  # Should have suggested but didn't
            success = False
            failure_reason = "Failed to suggest correction"

        # Store test results
        test_result = {
            "scenario": test_case.scenario,
            "query": test_case.query,
            "context": test_case.context,
            "expected": test_case.expected_correction,
            "got": result.correction if result.should_suggest else test_case.query,
            "confidence": result.confidence if result.should_suggest else 0.0,
            "success": success,
            "failure_reason": failure_reason,
            "processing_time": processing_time,
            "suggestion": self.system.format_response(result) if result.should_suggest else ""
        }

        # Group results by category
        if test_case.category not in self.results:
            self.results[test_case.category] = []
        self.results[test_case.category].append(test_result)

    def print_results(self):
        """Print formatted test results."""
        total_tests = sum(len(tests) for tests in self.results.values())
        successful_tests = sum(
            sum(1 for test in tests if test["success"])
            for tests in self.results.values()
        )

        print("\nSpell Correction System Test Results")
        print("=" * 100)
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.2f}%")
        print("=" * 100)

        for category, tests in self.results.items():
            print(f"\n{category.replace('_', ' ').title()} Tests:")
            print("-" * 100)

            for test in tests:
                print(f"\nScenario: {test['scenario']}")
                print(f"Query: {test['query']}")
                print(f"Context: {' '.join(test['context'])}")
                print(f"Result: {'✓' if test['success'] else '✗'}")

                if test['success']:
                    print(f"Confidence: {test['confidence']:.3f}")
                    if test['suggestion']:
                        print(f"Suggestion: {test['suggestion']}")
                else:
                    print(f"Failure Reason: {test['failure_reason']}")

                print(f"Processing Time: {test['processing_time']*1000:.2f}ms")
                print("-" * 50)

if __name__ == "__main__":
    # Run tests
    tester = SpellCorrectionTester("seed_data.json")
    tester.run_tests()
    tester.print_results()