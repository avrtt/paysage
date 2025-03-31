import unittest
import pandas as pd
import numpy as np
from gnomych.profiling import DataProfiler

class TestDataProfiler(unittest.TestCase):
    def setUp(self):
        data = {
            "A": [1, None, 3, 4, None],
            "B": [10, 20, 30, 40, 50],
            "C": [100, 200, 300, 400, 500]
        }
        self.df = pd.DataFrame(data)
        self.profiler = DataProfiler(self.df)

    def test_profile_missing(self):
        missing = self.profiler.profile_missing()
        self.assertEqual(missing["missing_summary"]["A"], 2)

    def test_profile_statistics(self):
        stats = self.profiler.profile_statistics()
        self.assertIn("A", stats)

    def test_profile_outliers(self):
        outliers = self.profiler.profile_outliers(method="iqr", threshold=1.5)
        self.assertIsInstance(outliers, dict)

if __name__ == '__main__':
    unittest.main()