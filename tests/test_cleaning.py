import unittest
import pandas as pd
import numpy as np
from gnomych.cleaning import DataCleaner

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame
        data = {
            "A": [1, 2, 2, 4, None],
            "B": ["x", "y", "y", "z", "x"],
            "C": [10, 20, 20, 40, 50]
        }
        self.df = pd.DataFrame(data)
        self.cleaner = DataCleaner(self.df)

    def test_remove_duplicates(self):
        df_cleaned = self.cleaner.remove_duplicates(subset=["B"])
        self.assertTrue(df_cleaned["B"].nunique() == len(df_cleaned))

    def test_fill_missing_values(self):
        df_filled = self.cleaner.fill_missing_values(strategy="mean", columns=["A"])
        self.assertFalse(df_filled["A"].isnull().any())

    def test_standardize_columns(self):
        df_std = self.cleaner.standardize_columns()
        for col in df_std.columns:
            self.assertTrue(col.islower())
            self.assertNotIn(" ", col)

    def test_detect_outliers(self):
        # Create a DataFrame with an outlier
        df_test = pd.DataFrame({"X": [1, 2, 3, 4, 1000]})
        cleaner = DataCleaner(df_test)
        outliers = cleaner.detect_outliers(method="zscore", threshold=2.0, columns=["X"])
        self.assertTrue(len(outliers["X"]) > 0)

if __name__ == '__main__':
    unittest.main()