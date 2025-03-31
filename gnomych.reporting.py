import logging
import json
from datetime import datetime
import os

class ReportGenerator:
    """
    ReportGenerator generates reports from data profiling and validation data.
    Supports markdown and HTML report generation.
    """
    def __init__(self, report_data: dict, output_dir: str = "reports"):
        self.report_data = report_data
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info("Created output directory: %s", self.output_dir)
    
    def generate_markdown_report(self) -> str:
        """
        Generate a markdown formatted report.
        
        Returns:
        - A string containing the markdown report.
        """
        self.logger.info("Generating markdown report.")
        lines = []
        lines.append(f"# Data Profiling Report")
        lines.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## Missing Values")
        for col, val in self.report_data.get("missing", {}).get("missing_summary", {}).items():
            lines.append(f"- **{col}**: {val} missing")
        lines.append("\n## Summary Statistics")
        statistics = self.report_data.get("statistics", {})
        for col, stats in statistics.items():
            lines.append(f"### {col}")
            for stat, value in stats.items():
                lines.append(f"- {stat}: {value}")
        lines.append("\n## Outliers")
        outliers = self.report_data.get("outliers", {})
        for col, data in outliers.items():
            lines.append(f"- **{col}**: {data.get('count', 0)} outliers")
        markdown_report = "\n".join(lines)
        self.logger.info("Markdown report generated.")
        return markdown_report

    def generate_html_report(self) -> str:
        """
        Generate an HTML formatted report.
        
        Returns:
        - A string containing the HTML report.
        """
        self.logger.info("Generating HTML report.")
        html = []
        html.append(f"<html><head><title>Data Profiling Report</title></head><body>")
        html.append(f"<h1>Data Profiling Report</h1>")
        html.append(f"<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<h2>Missing Values</h2><ul>")
        for col, val in self.report_data.get("missing", {}).get("missing_summary", {}).items():
            html.append(f"<li><strong>{col}:</strong> {val} missing</li>")
        html.append("</ul>")
        html.append("<h2>Summary Statistics</h2>")
        statistics = self.report_data.get("statistics", {})
        for col, stats in statistics.items():
            html.append(f"<h3>{col}</h3><ul>")
            for stat, value in stats.items():
                html.append(f"<li>{stat}: {value}</li>")
            html.append("</ul>")
        html.append("<h2>Outliers</h2><ul>")
        outliers = self.report_data.get("outliers", {})
        for col, data in outliers.items():
            html.append(f"<li><strong>{col}:</strong> {data.get('count', 0)} outliers</li>")
        html.append("</ul></body></html>")
        html_report = "\n".join(html)
        self.logger.info("HTML report generated.")
        return html_report

    def save_report(self, report_str: str, filename: str):
        """
        Save the report string to a file within the output directory.
        
        Parameters:
        - report_str: The report content as a string.
        - filename: The name of the file to save (e.g., 'report.md').
        """
        filepath = os.path.join(self.output_dir, filename)
        self.logger.info("Saving report to %s", filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_str)
        self.logger.info("Report saved successfully at %s", filepath)