import pandas as pd
import jsonschema
import logging
from gnomych.exceptions import DataValidationError

class DataValidator:
    """
    DataValidator validates a pandas DataFrame against a given JSON schema and custom business rules.
    """
    def __init__(self, df: pd.DataFrame, schema: dict = None):
        self.df = df.copy()
        self.schema = schema
        self.logger = logging.getLogger(__name__)
        self.rules = []
    
    def load_schema(self, schema: dict):
        """
        Load JSON schema for validation.
        
        Parameters:
        - schema: a dictionary representing the JSON schema
        """
        self.schema = schema
        self.logger.info("Schema loaded with keys: %s", list(schema.keys()))
    
    def add_rule(self, rule_func):
        """
        Add a custom rule function to the validator.
        
        Each rule function should accept a row (pandas Series) and return True if the row passes.
        """
        self.rules.append(rule_func)
        self.logger.info("Added custom rule: %s", rule_func.__name__)
    
    def validate_schema(self):
        """
        Validate each row of the DataFrame against the JSON schema.
        
        Returns:
        - A dictionary with row indices as keys and validation error messages as values.
        """
        if self.schema is None:
            self.logger.error("No schema defined for validation.")
            raise DataValidationError("Schema is not defined.")
        errors = {}
        for index, row in self.df.iterrows():
            data_dict = row.to_dict()
            try:
                jsonschema.validate(instance=data_dict, schema=self.schema)
            except jsonschema.ValidationError as e:
                errors[index] = str(e)
                self.logger.debug("Validation error at index %d: %s", index, e)
        if errors:
            self.logger.warning("Schema validation found errors in %d rows", len(errors))
        return errors
    
    def validate_rules(self):
        """
        Apply custom business rules to the DataFrame.
        
        Returns:
        - A dictionary mapping row indices to a list of rules that failed.
        """
        errors = {}
        for rule in self.rules:
            for index, row in self.df.iterrows():
                try:
                    result = rule(row)
                    if not result:
                        if index not in errors:
                            errors[index] = []
                        errors[index].append(rule.__name__)
                        self.logger.debug("Rule %s failed at index %d", rule.__name__, index)
                except Exception as e:
                    self.logger.error("Error executing rule %s at index %d: %s", rule.__name__, index, e)
                    if index not in errors:
                        errors[index] = []
                    errors[index].append(f"{rule.__name__}: exception {str(e)}")
        if errors:
            self.logger.warning("Custom rule validation found errors in %d rows", len(errors))
        return errors
    
    def validate(self):
        """
        Run full validation: first check schema compliance, then custom business rules.
        
        Returns:
        - A combined dictionary with row indices as keys and a dictionary of errors as values.
        """
        self.logger.info("Starting full validation.")
        schema_errors = self.validate_schema()
        rule_errors = self.validate_rules()
        all_errors = {}
        for index in set(list(schema_errors.keys()) + list(rule_errors.keys())):
            all_errors[index] = {
                "schema": schema_errors.get(index, None),
                "rules": rule_errors.get(index, None)
            }
        self.logger.info("Validation complete with %d errors", len(all_errors))
        return all_errors