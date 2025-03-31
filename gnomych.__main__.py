import argparse
import pandas as pd
import logging
from gnomych.cleaning import DataCleaner
from gnomych.profiling import DataProfiler
from gnomych.reporting import ReportGenerator
from gnomych.utils import configure_logging

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Data Cleaning and Profiling Tool")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--report", required=True, help="Report output file (markdown)")
    args = parser.parse_args()
    
    logging.info("Reading input CSV file: %s", args.input)
    df = pd.read_csv(args.input)
    
    logging.info("Starting data cleaning process.")
    cleaner = DataCleaner(df)
    df_clean, outliers = cleaner.clean_data()
    
    logging.info("Generating data profile report.")
    profiler = DataProfiler(df_clean)
    report_data = profiler.generate_profile_report()
    
    reporter = ReportGenerator(report_data)
    md_report = reporter.generate_markdown_report()
    reporter.save_report(md_report, args.report)
    logging.info("Report generated successfully at: %s", args.report)
    print("Report generated successfully.")

if __name__ == "__main__":
    main()