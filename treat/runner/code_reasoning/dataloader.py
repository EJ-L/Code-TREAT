"""
Data loading utilities for code reasoning tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import json
from .data import Data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GFG_DATA_DIRS = [
    os.path.join(PROJECT_ROOT, "..", "data", "geeksforgeeks", "code_execution", "java.jsonl"), 
    os.path.join(PROJECT_ROOT, "..", "data", "geeksforgeeks", "code_execution", "python.jsonl")
]
HR_DATA_DIRS = [
    os.path.join(PROJECT_ROOT, "..", "data", "hackerrank", "code_execution", "java.jsonl"), 
    os.path.join(PROJECT_ROOT, "..", "data", "hackerrank", "code_execution", "python.jsonl")
]

class DataLoader:
    """Data loader for code reasoning with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str):
        """Initialize the data loader with dataset path"""
        self.dataset = dataset
        self.language = language


    def load_data(self):
        """Load all code reasoning data from the dataset"""
        if self.dataset == 'geeksforgeeks':
            return self.load_gfg()
        if self.dataset == 'hackerrank':
            return self.load_hr()
            
    def load_hr(self):
        """Load HackerRank data"""
        organized_data = []
        for path in HR_DATA_DIRS:
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]
                for item in data:
                    _id = item['question_id']
                    difficulty = item['difficulty']
                    masked_test_code = item['masked_test_code']
                    # for test_case in item['test_cases']:
                    organized_data.append(Data(
                        _id=_id,
                        dataset=self.dataset,
                        difficulty=difficulty,
                        language=self.language,
                        function=masked_test_code,
                        test_case_info=item['test_cases']
                    ))
        return organized_data
    
    def load_gfg(self):
        """Load PolyHumanEval data"""
        organized_data = []
        for path in GFG_DATA_DIRS:
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]
                for item in data:
                    _id = item['question_id']
                    difficulty = item['difficulty']
                    masked_test_code = item['masked_test_code']
                    _id = item['question_id']
                    difficulty = item['difficulty']
                    masked_test_code = item['masked_test_code']
                    # for test_case in item['test_cases']:
                    organized_data.append(Data(
                        _id=_id,
                        dataset=self.dataset,
                        difficulty=difficulty,
                        language=self.language,
                        function=masked_test_code,
                        test_case_info=item['test_cases']
                    ))
        return organized_data