import os
import json
import logging
from typing import Dict

def setup_logging(log_file_path: str):
    """
    Sets up logging to file and console.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

def save_token_usage(token_usage: Dict[str, int], file_path: str):
    """
    Saves or updates token usage in a JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                current_usage = json.load(f)
        except json.JSONDecodeError:
            pass # Start fresh if file is corrupted

    # Accumulate usage
    for key in current_usage:
        current_usage[key] += token_usage.get(key, 0)

    with open(file_path, 'w') as f:
        json.dump(current_usage, f, indent=4)
