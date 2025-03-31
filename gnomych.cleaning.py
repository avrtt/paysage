import pandas as pd
import numpy as np
import logging

class DataCleaner:
    """
    DataCleaner performs various data cleaning operations on a pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataCleaner initialized with dataframe of shape %s", self.df.shape)
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows based on a subset of columns."""
        self.logger.info("Removing duplicates with subset: %s", subset)
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = self.df.shape[0]
        self.logger.info("Removed %d duplicate rows", before - after)
        return self.df

    def fill_missing_values(self, strategy='mean', columns=None, fill_value=None):
        """
        Fill missing values using the specified strategy.
        
        Parameters:
        - strategy: 'mean', 'median', 'mode', or 'constant'
        - columns: list of columns to process (defaults to all)
        - fill_value: the constant value to use when strategy is 'constant'
        """
        self.logger.info("Filling missing values with strategy: %s", strategy)
        if columns is None:
            columns = self.df.columns.tolist()
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean' and self.df[col].dtype in [np.float64, np.int64]:
                    value = self.df[col].mean()
                elif strategy == 'median' and self.df[col].dtype in [np.float64, np.int64]:
                    value = self.df[col].median()
                elif strategy == 'mode':
                    value = self.df[col].mode()[0]
                elif strategy == 'constant':
                    value = fill_value
                else:
                    self.logger.warning("Unsupported strategy for column %s", col)
                    continue
                self.logger.debug("Filling column %s with value %s", col, value)
                self.df[col].fillna(value, inplace=True)
        return self.df

    def standardize_columns(self):
        """
        Standardize column names: strip whitespace, convert to lowercase,
        and replace spaces with underscores.
        """
        self.logger.info("Standardizing column names.")
        new_columns = {}
        for col in self.df.columns:
            new_col = col.strip().lower().replace(' ', '_')
            new_columns[col] = new_col
        self.df.rename(columns=new_columns, inplace=True)
        self.logger.debug("Columns renamed: %s", new_columns)
        return self.df

    def detect_outliers(self, method='zscore', threshold=3.0, columns=None):
        """
        Detect outliers in numerical columns using the specified method.
        
        Parameters:
        - method: 'zscore' or 'iqr'
        - threshold: threshold for determining outliers
        - columns: specific columns to check (defaults to all numeric)
        
        Returns:
        - A dictionary mapping column names to a list of row indices considered as outliers.
        """
        self.logger.info("Detecting outliers using %s method", method)
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_indices = {}
        if method == 'zscore':
            for col in columns:
                col_mean = self.df[col].mean()
                col_std = self.df[col].std()
                self.logger.debug("Column %s: mean=%s, std=%s", col, col_mean, col_std)
                z_scores = (self.df[col] - col_mean) / col_std
                outliers = self.df.index[np.abs(z_scores) > threshold].tolist()
                outlier_indices[col] = outliers
                self.logger.info("Detected %d outliers in column %s", len(outliers), col)
        elif method == 'iqr':
            for col in columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df.index[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].tolist()
                outlier_indices[col] = outliers
                self.logger.info("Detected %d outliers in column %s", len(outliers), col)
        else:
            self.logger.error("Unsupported outlier detection method: %s", method)
            raise ValueError(f"Unsupported method: {method}")
        return outlier_indices

    def clean_data(self):
        """
        Run the full data cleaning pipeline:
         - Standardize column names
         - Remove duplicates
         - Fill missing values using the median strategy
         - Detect outliers using the IQR method
        
        Returns:
         - Cleaned DataFrame and dictionary of outlier indices.
        """
        self.logger.info("Starting full data cleaning pipeline.")
        self.standardize_columns()
        self.remove_duplicates()
        self.fill_missing_values(strategy='median')
        outliers = self.detect_outliers(method='iqr')
        self.logger.info("Data cleaning pipeline complete.")
        return self.df, outliers