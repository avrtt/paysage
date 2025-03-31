import unittest
import os
from gnomych.reporting import ReportGenerator

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.report_data = {
            "missing": {"missing_summary": {"A": 2, "B": 0}},
            "statistics": {"A": {"mean": 3, "std": 1.414}, "B": {"mean": 30, "std": 15.811}},
            "outliers": {"A": {"count": 0, "indices": []}, "B": {"count": 0, "indices": []}}
        }
        self.generator = ReportGenerator(self.report_data, output_dir="test_reports")

    def test_markdown_report(self):
        md = self.generator.generate_markdown_report()
        self.assertIn("Data Profiling Report", md)

    def test_html_report(self):
        html = self.generator.generate_html_report()
        self.assertIn("<html>", html)

    def test_save_report(self):
        md = self.generator.generate_markdown_report()
        self.generator.save_report(md, "test_report.md")
        self.assertTrue(os.path.exists("test_reports/test_report.md"))
        # Cleanup
        os.remove("test_reports/test_report.md")
        os.rmdir("test_reports")

if __name__ == '__main__':
    unittest.main()