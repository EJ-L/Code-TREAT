from pathlib import Path
import json
from .data import CodeSumData
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset: str, language: str,):
        self.dataset = dataset
        self.language = language
        
    def load_data(self):
        organized_data = []
        ds = load_dataset("Code-TREAT/code_summarization")
        full_data = ds['test']
        for data in full_data:
            if data['dataset'] != self.dataset or data['lang'] != self.language:
                continue
            organized_data.append(CodeSumData(data))