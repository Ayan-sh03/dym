import json
from typing import Dict, List, Optional
import re

class DataPreprocessor:
    def __init__(self, seed_data_path: str):
        """Initialize the preprocessor with seed data."""
        self.seed_data = self._load_seed_data(seed_data_path)
        self.user_history: Dict[str, int] = {}  # Simple user query history

    def _load_seed_data(self, path: str) -> dict:
        """Load seed data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the input query:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Remove special characters except apostrophes
        4. Normalize text
        """
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Remove special characters except apostrophes
        query = re.sub(r'[^a-z0-9\s\']', '', query)

        return query

    def tokenize(self, query: str) -> List[str]:
        """Split query into tokens."""
        return query.split()

    def extract_user_profile(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Extract user-specific data for personalization.
        Returns user history or empty dict if no history exists.
        """
        if user_id and user_id in self.user_history:
            return self.user_history[user_id]
        return {}

    def update_user_history(self, user_id: str, query: str) -> None:
        """Update user's search history."""
        if user_id not in self.user_history:
            self.user_history[user_id] = {}

        tokens = self.tokenize(query)
        for token in tokens:
            self.user_history[user_id][token] = self.user_history[user_id].get(token, 0) + 1

    def process_query(self, query: str, user_id: Optional[str] = None) -> tuple:
        """
        Complete query processing pipeline.
        Returns (preprocessed_query, tokens, user_profile)
        """
        # Preprocess the query
        processed_query = self.preprocess_query(query)

        # Tokenize
        tokens = self.tokenize(processed_query)

        # Get user profile
        user_profile = self.extract_user_profile(user_id)

        # Update user history if user_id provided
        if user_id:
            self.update_user_history(user_id, processed_query)

        return processed_query, tokens, user_profile

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor("seed_data.json")

    # Test query
    test_query = "WeaTHer forecst 2day!!"
    user_id = "test_user_1"

    # Process query
    processed_query, tokens, user_profile = preprocessor.process_query(test_query, user_id)

    print(f"Original query: {test_query}")
    print(f"Processed query: {processed_query}")
    print(f"Tokens: {tokens}")
    print(f"User profile: {user_profile}")