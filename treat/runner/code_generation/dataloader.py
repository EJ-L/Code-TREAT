"""
Data loading utilities for code translation tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import json
from .data import HackerrankData, GeeksforGeeksData
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GFG_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "geeksforgeeks", "geeksforgeeks_filtered_valid.jsonl")
HR_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "hackerrank", "hackerrank_filtered.jsonl")

class DataLoader:
    """Data loader for code translation with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str):
        """Initialize the data loader with dataset path"""
        self.dataset = dataset
        self.language = language


    def load_data(self):
        """Load all code translation data from the dataset"""
        if self.dataset == 'geeksforgeeks':
            return self.load_gfg()
        if self.dataset == 'hackerrank':
            return self.load_hr()
            
    def load_hr(self):
        """Load HackerRank data"""
        organized_data = []
        with open(HR_DATA_DIR, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
            for item in data:
                problem_description = item['question_content']
                if item.get('input_format', ""):
                    problem_description += '\n' + item['input_format']
                if item.get('output_format', ""):
                    problem_description += '\n' + item['output_format']
                
                language_metadata = item[self.language]
                template_head = language_metadata.get('template_head', '').strip()
                template_body = language_metadata.get('template', '').strip()
                template_tail = language_metadata.get('template_tail', '').strip()
                starter_code = template_head + '\n' + template_body + '\n' + template_tail
                class_name = language_metadata.get('class_name', None)
                func_sign = language_metadata.get('func_sign', None)
                
                organized_data.append(HackerrankData(
                    id=item['question_id'],
                    title=item['question_title'],
                    problem_description=problem_description,
                    difficulty=item['difficulty'],
                    release_date=item['release_date'],
                    func_sign=func_sign,
                    driver_code=template_tail,
                    starter_code=starter_code,
                    class_name=class_name,
                ))
        return organized_data
    
    def load_gfg(self):
        """Load PolyHumanEval data"""
        with open(GFG_DATA_DIR, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
            data = [GeeksforGeeksData(
                id=item['question_id'],
                question_title=item['question_title'],
                problem_description=item['question_content'],
                difficulty=item['difficulty'],
                release_date=item['release_date'],
                func_sign=item[self.language]['func_sign'],
                driver_code=item[self.language]['initial_code'],
                starter_code=item[self.language]['initial_code'] + '\n' + item[self.language]['user_code'],
                class_name=item[self.language].get('class_name', None),
            ) for item in data]
        return data