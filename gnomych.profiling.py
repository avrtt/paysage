import pandas as pd
import numpy as np
import logging

class DataProfiler:
    """
    DataProfiler provides methods to generate profiling reports for a pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)
    
    def profile_missing(self):
        """
        Profile missing values in the DataFrame.
        
        Returns:
        - A dictionary with the count of missing values per column and the total missing count.
        """
        self.logger.info("Profiling missing values.")
        missing_summary = self.df.isnull().sum().to_dict()
        total_missing = sum(missing_summary.values())
        self.logger.debug("Missing summary: %s", missing_summary)
        return {"missing_summary": missing_summary, "total_missing": total_missing}
    
    def profile_statistics(self):
        """
        Generate summary statistics for numerical columns.
        
        Returns:
        - A dictionary of descriptive statistics.
        """
        self.logger.info("Generating summary statistics.")
        desc = self.df.describe().to_dict()
        self.logger.debug("Summary statistics: %s", desc)
        return desc
    
    def profile_outliers(self, method='iqr', threshold=1.5):
        """
        Profile outliers in numerical columns using the IQR method.
        
        Parameters:
        - method: currently only supports 'iqr'
        - threshold: multiplier for the IQR
        
        Returns:
        - A dictionary mapping each numeric column to outlier counts and indices.
        """
        self.logger.info("Profiling outliers using %s method", method)
        outlier_summary = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            series = self.df[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_summary[col] = {
                "count": outliers.count(),
                "indices": outliers.index.tolist()
            }
            self.logger.debug("Column %s has %d outliers", col, outliers.count())
        return outlier_summary
    
    def generate_profile_report(self):
        """
        Generate a comprehensive data profile report that includes missing values,
        summary statistics, and outlier information.
        
        Returns:
        - A dictionary with all profile sections.
        """
        self.logger.info("Generating full profile report.")
        report = {
            "missing": self.profile_missing(),
            "statistics": self.profile_statistics(),
            "outliers": self.profile_outliers()
        }
        self.logger.info("Profile report generated.")
        return report