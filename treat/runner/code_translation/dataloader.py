"""
Data loading utilities for code translation tasks
"""
from typing import Dict, List, Any, Optional, Iterator, Tuple
import os
import json
from .data import HackerrankData, PolyHumanEvalData
from datasets import load_dataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLY_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "polyhumaneval", "polyhumaneval_sol.json")

class DataLoader:
    """Data loader for code translation with batching and filtering capabilities"""
    
    def __init__(self, dataset: str, language: str):
        """Initialize the data loader with dataset path"""
        # assert the language is outputted in the format of SL->TL
        self.dataset = dataset
        self.source_lang, self.target_lang = language.split('->')

    def load_data(self):
        organized_data = []
        ds = load_dataset('Code-TREAT/code_translation')
        full_data = ds['test']
        for data in full_data:
            if data['language'] != self.source_lang or data['dataset'] != self.dataset:
                continue
            if data['dataset'] == 'hackerrank':
                organized_data.append(HackerrankData(
                    id=data['question_id'],
                    title=data['question_title'],
                    difficulty=data['difficulty'],
                    domain=data.get("domain", "hr"),
                    release_date=data['release_date'],
                    source_code=data[self.source_lang]['solution'],
                ))
            elif data['dataset'] == 'polyhumaneval':
                organized_data.append(PolyHumanEvalData(
                    id=data['question_id'],
                    source_code=data[self.source_lang]['solution'],
                ))
        return organized_data