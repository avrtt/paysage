import logging
import os
import json
import datetime

def configure_logging(log_level=logging.DEBUG, log_file=None):
    """
    Configure logging for the application.
    
    Parameters:
    - log_level: Logging level (e.g., logging.DEBUG)
    - log_file: Optional file to log messages into
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_config(config_path: str) -> dict:
    """
    Load JSON configuration from a file.
    
    Parameters:
    - config_path: Path to the configuration file
    
    Returns:
    - A dictionary with configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def save_config(config: dict, config_path: str):
    """
    Save a configuration dictionary to a JSON file.
    
    Parameters:
    - config: The configuration dictionary
    - config_path: The path to save the JSON file
    """
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def get_timestamp() -> str:
    """
    Return the current timestamp in ISO format.
    """
    return datetime.datetime.now().isoformat()