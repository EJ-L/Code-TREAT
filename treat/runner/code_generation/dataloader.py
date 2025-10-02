"""
Data loading utilities for code generation tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import json
from .data import Data
from datasets import load_dataset

class DataLoader:
    """Data loader for code generation with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str, reproduce: bool = True):
        """Initialize the data loader with dataset path"""
        self.dataset = dataset
        self.language = language
        self.reproduce = reproduce

    def load_data(self):
        """Load all code generation data from the dataset"""
        dataset_name = "Code-TREAT/code_generation_lite" if self.reproduce else "Code-TREAT/code_generation"
        if self.dataset not in {"geeksforgeeks", "hackerrank"}:
            raise ValueError(f"Unknown dataset {self.dataset}")
        ds = load_dataset(dataset_name)
        full_data = ds['test']
        organized_data = []
        for data in full_data:
            dataset = data['dataset']
            if dataset != self.dataset:
                continue
            _id = data['question_id']
            question_title=data['question_title']
            problem_description=data['question_content']
            difficulty=data['difficulty']
            release_date=data.get('release_date','')
            language_metadata = data[self.language]

            if dataset == 'hackerrank':
                template_head = language_metadata.get('template_head', '').strip()
                template_body = language_metadata.get('template', '').strip()
                driver_code = language_metadata.get('template_tail', '').strip()
                starter_code = template_head + '\n' + template_body + '\n' + driver_code
                class_name = language_metadata.get('class_name', None)
                func_sign = language_metadata.get('func_sign', None)            
            elif dataset == 'geeksforgeeks':
                func_sign=language_metadata['func_sign'],
                driver_code=language_metadata['initial_code'],
                starter_code=language_metadata['initial_code'] + '\n' + language_metadata['user_code'],
                class_name=language_metadata.get('class_name', None),
            organized_data.append(Data(
                id=_id,
                dataset=dataset,
                title=question_title,
                problem_description=problem_description,
                difficulty=difficulty,
                release_date=release_date,
                func_sign=func_sign,
                driver_code=driver_code,
                starter_code=starter_code,
                class_name=class_name,
            ))
        return organized_data
