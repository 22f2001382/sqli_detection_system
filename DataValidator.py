import pandas as pd
import re

class DataValidator:

    SQL_PATTERN = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|UNION|JOIN)\b", re.IGNORECASE)
    REQUIRED_COLUMNS = {'query', 'label'}

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {str(e)}")

    def validate_columns(self):
        if set(self.df.columns) != self.REQUIRED_COLUMNS:
            return False, "CSV must contain exactly two columns: 'query' and 'label'."
        return True, "Columns valid."

    def validate_labels(self):
        if self.df['label'].isnull().any():
            return False, "Missing labels detected."
        if not set(self.df['label'].unique()).issubset({0, 1}):
            return False, "Labels must be either 0 (benign) or 1 (malicious)."
        return True, "Labels valid."

    def validate_queries(self):
        """Ensure each query is a valid SQL statement."""
        for query in self.df['query']:
            if not isinstance(query, str) or not self.SQL_PATTERN.search(query):
                return False, f"Invalid query detected: {query}"
        return True, "Queries valid."

    def run_validations(self):
        validations = [
            self.validate_columns(),
            self.validate_labels(),
            self.validate_queries()
        ]

        for valid, message in validations:
            if not valid:
                return False, message  # Return first encountered error

        return True, "CSV passed all validation checks."

# Example usage
