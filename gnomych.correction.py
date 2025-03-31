import pandas as pd
import numpy as np
import logging

class DataCorrectionSuggester:
    """
    DataCorrectionSuggester provides suggestions and automated corrections for common data issues.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)
    
    def suggest_imputation(self, column: str):
        """
        Suggest an imputation strategy for a given column.
        
        For numerical columns, returns the mean; for categorical, returns the mode.
        
        Returns:
        - A dictionary with the suggested strategy and value.
        """
        self.logger.info("Suggesting imputation for column %s", column)
        if self.df[column].dtype in [np.float64, np.int64]:
            suggestion = {
                "strategy": "mean",
                "value": self.df[column].mean()
            }
        else:
            suggestion = {
                "strategy": "mode",
                "value": self.df[column].mode()[0] if not self.df[column].mode().empty else None
            }
        self.logger.debug("Imputation suggestion for %s: %s", column, suggestion)
        return suggestion

    def suggest_outlier_handling(self, column: str, method='iqr', threshold=1.5):
        """
        Suggest a method to handle outliers in a given column.
        
        Returns:
        - A dictionary with lower/upper bounds and the recommended action.
        """
        self.logger.info("Suggesting outlier handling for column %s", column)
        series = self.df[column]
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            suggestion = {
                "method": "iqr",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "action": "clip"
            }
        elif method == 'zscore':
            mean_val = series.mean()
            std_val = series.std()
            suggestion = {
                "method": "zscore",
                "mean": mean_val,
                "std": std_val,
                "threshold": 3.0,
                "action": "remove or clip"
            }
        else:
            self.logger.error("Unsupported method for outlier handling: %s", method)
            suggestion = {"error": "Unsupported method"}
        self.logger.debug("Outlier handling suggestion for %s: %s", column, suggestion)
        return suggestion

    def auto_correct(self, corrections: dict):
        """
        Automatically apply corrections to the DataFrame.
        
        The corrections dictionary should map column names to correction instructions.
        For example:
            {
              "A": {"action": "impute"},
              "B": {"action": "clip", "method": "iqr"}
            }
        
        Returns:
        - The corrected DataFrame.
        """
        self.logger.info("Starting auto-correction.")
        for col, instruction in corrections.items():
            if instruction.get("action") == "impute":
                self.logger.info("Auto-correct: imputing missing values for %s", col)
                suggestion = self.suggest_imputation(col)
                self.df[col].fillna(suggestion.get("value"), inplace=True)
            elif instruction.get("action") == "clip":
                self.logger.info("Auto-correct: clipping outliers for %s", col)
                suggestion = self.suggest_outlier_handling(col, method=instruction.get("method", "iqr"))
                lower = suggestion.get("lower_bound")
                upper = suggestion.get("upper_bound")
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            else:
                self.logger.warning("No valid action specified for column %s", col)
        self.logger.info("Auto-correction complete.")
        return self.df