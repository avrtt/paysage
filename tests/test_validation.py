import unittest
import pandas as pd
from gnomych.validation import DataValidator
from gnomych.exceptions import DataValidationError

def rule_positive(row):
    return row["value"] > 0

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        data = {"value": [10, -5, 15]}
        self.df = pd.DataFrame(data)
        self.schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        }
        self.validator = DataValidator(self.df, self.schema)
        self.validator.add_rule(rule_positive)

    def test_validate_schema(self):
        errors = self.validator.validate_schema()
        self.assertEqual(len(errors), 0)

    def test_validate_rules(self):
        errors = self.validator.validate_rules()
        self.assertIn(1, errors)  # The -5 should trigger the rule
        self.assertIn("rule_positive", errors[1])

    def test_full_validation(self):
        errors = self.validator.validate()
        self.assertIn(1, errors)
        self.assertIn("rule_positive", errors[1]["rules"])

if __name__ == '__main__':
    unittest.main()