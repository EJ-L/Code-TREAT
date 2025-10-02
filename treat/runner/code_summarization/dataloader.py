from pathlib import Path
import json
from .data import CodeSumData
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset: str, language: str, reproduce: bool = True):
        self.dataset = dataset
        self.language = language
        self.reproduce = reproduce
        
    def load_data(self):
        organized_data = []
        dataset_name = "Code-TREAT/code_summarization_lite" if self.reproduce else "Code-TREAT/code_summarization"
        ds = load_dataset(dataset_name)
        full_data = ds['test']
        for data in full_data:
            if data['dataset'] != self.dataset or data['lang'] != self.language:
                continue
            organized_data.append(CodeSumData(data, default_dataset=self.dataset))
        return organized_data
