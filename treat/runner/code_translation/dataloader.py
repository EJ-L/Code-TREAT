"""
Data loading utilities for code translation tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import json
from .data import HackerrankData, PolyHumanEvalData
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HR_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "hackerrank", "hackerrank_filtered.jsonl")
POLY_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "polyhumaneval", "polyhumaneval_sol.json")

class DataLoader:
    """Data loader for code translation with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str):
        """Initialize the data loader with dataset path"""
        # assert the language is outputted in the format of SL->TL
        self.source_lang, self.target_lang = language.split('->')
        self.dataset = dataset

    def load_data(self):
        """Load all code translation data from the dataset"""
        if self.dataset == 'polyhumaneval':
            return self.load_poly()
        if self.dataset == 'hackerrank':
            return self.load_hr()
            
    def load_hr(self):
        """Load HackerRank data"""
        with open(HR_DATA_DIR, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
            data = [HackerrankData(
                id=item['question_id'],
                title=item['question_title'],
                difficulty=item['difficulty'],
                domain=item.get("domain", "hr"),
                release_date=item['release_date'],
                source_code=item[self.source_lang]['solution'],
            ) for item in data]
        return data
    
    def load_poly(self):
        """Load PolyHumanEval data"""
        with open(POLY_DATA_DIR, 'r') as f:
            data = json.load(f)
            data = [PolyHumanEvalData(
                id=_id,
                source_code=sol,
            ) for _id, sol in data[self.source_lang].items()]
        return data