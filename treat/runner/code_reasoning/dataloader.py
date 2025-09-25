"""
Data loading utilities for code reasoning tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import json
from .data import Data
from datasets import load_dataset

class DataLoader:
    """Data loader for code reasoning with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str):
        """Initialize the data loader with dataset path"""
        self.dataset = dataset
        self.language = language
    
    def load_data(self):
        ds = load_dataset("Code-TREAT/code_reasoning")
        full_data = ds['test']
        organized_data = []
        for data in full_data:
            dataset = data['dataset']
            lang = data['lang']
            if lang != self.language or dataset != self.dataset:
                continue
            _id = data['question_id']
            difficulty = data['difficulty']
            masked_test_code = data['masked_test_code']
            organized_data.append(Data(
                _id=_id,
                dataset=self.dataset,
                difficulty=difficulty,
                language=self.language,
                function=masked_test_code,
                test_case_info=data['test_cases']
            ))
        return organized_data