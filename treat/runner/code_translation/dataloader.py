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
    
    def __init__(self, dataset: str, language: str, reproduce: bool = True):
        """Initialize the data loader with dataset path"""
        # assert the language is outputted in the format of SL->TL
        self.dataset = dataset
        self.source_lang, self.target_lang = language.split('->')
        self.reproduce = reproduce

    def load_data(self):
        organized_data = []
        if self.dataset == 'hackerrank':
            dataset_name = 'Code-TREAT/code_translation_lite' if self.reproduce else 'Code-TREAT/code_translation'
            ds = load_dataset(dataset_name)
            full_data = ds['test']
            for data in full_data:
                if data['dataset'] == 'hackerrank':
                    organized_data.append(HackerrankData(
                        id=data['question_id'],
                        title=data['question_title'],
                        difficulty=data['difficulty'],
                        domain=data.get("domain", "hr"),
                        release_date="",
                        source_code=data[self.source_lang]['solution'],
                    ))
        elif self.dataset == 'polyhumaneval':
            with open(POLY_DATA_DIR, 'r') as f:
                data = json.load(f)
                organized_data = [PolyHumanEvalData(
                    id=_id,
                    source_code=sol,
                ) for _id, sol in data[self.source_lang].items()]
        return organized_data
