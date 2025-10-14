"""
Utility functions for file operations with proper encoding handling.
"""

import json
import os
import yaml

def safe_load_json(file_path):
    """
    Safely load a JSON file with proper encoding handling.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        # Try with UTF-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except UnicodeDecodeError:
        # Fallback to reading with errors='ignore' if UTF-8 fails
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            return json.loads(content)

def safe_load_yaml(file_path):
    """
    Safely load a YAML file with proper encoding handling.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Loaded YAML data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    try:
        # Try with UTF-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except UnicodeDecodeError:
        # Fallback to reading with errors='ignore' if UTF-8 fails
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            return yaml.safe_load(content)

def safe_write_json(file_path, data, indent=2):
    """
    Safely write JSON data to a file with UTF-8 encoding.
    
    Args:
        file_path (str): Path to the JSON file
        data (dict): Data to write
        indent (int): Indentation level for pretty printing
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

def safe_write_yaml(file_path, data):
    """
    Safely write YAML data to a file with UTF-8 encoding.
    
    Args:
        file_path (str): Path to the YAML file
        data (dict): Data to write
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)
