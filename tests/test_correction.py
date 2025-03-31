import unittest
import pandas as pd
import numpy as np
from gnomych.correction import DataCorrectionSuggester

class TestDataCorrectionSuggester(unittest.TestCase):
    def setUp(self):
        data = {
            "A": [1, None, 3, None, 5],
            "B": [10, 20, 3000, 40, 50]
        }
        self.df = pd.DataFrame(data)
        self.suggester = DataCorrectionSuggester(self.df)

    def test_suggest_imputation(self):
        suggestion = self.suggester.suggest_imputation("A")
        self.assertEqual(suggestion["strategy"], "mean")

    def test_suggest_outlier_handling(self):
        suggestion = self.suggester.suggest_outlier_handling("B", method="iqr")
        self.assertIn("lower_bound", suggestion)
        self.assertIn("upper_bound", suggestion)

    def test_auto_correct(self):
        corrections = {
            "A": {"action": "impute"},
            "B": {"action": "clip", "method": "iqr"}
        }
        df_corrected = self.suggester.auto_correct(corrections)
        self.assertFalse(df_corrected["A"].isnull().any())
        # Check if B has been clipped within the suggested bounds
        suggestion = self.suggester.suggest_outlier_handling("B", method="iqr")
        self.assertTrue(df_corrected["B"].max() <= suggestion["upper_bound"])

if __name__ == '__main__':
    unittest.main()