"""
Data loading utilities for code review generation tasks
"""
import json
import os
from tqdm import tqdm
from .data import CodeReviewData
from typing import List
from datasets import load_dataset


class DataLoader:
    def __init__(self, dataset: str, language: str, reproduce: bool = True):
        self.dataset = dataset
        self.language = language
        self.reproduce = reproduce
        
    def load_data(self):
        data = []
        dataset_name = "Code-TREAT/code_review_generation_lite" if self.reproduce else "Code-TREAT/code_review_generation"
        ds = load_dataset(dataset_name)
        test_data = ds['test']
        for record in test_data:
            lang = record["lang"]
            dataset = record['dataset']
            if lang != self.language or dataset != self.dataset:
                continue
            owner = record["owner"]
            repo_name = record["repo_name"]
            pr_id = record["pr_id"]
            diff_hunk = record["diff_hunk"]
            reviewer = record["reviewer"]
            code_review_comment = record["code_review_comment"]
            data.append(CodeReviewData(
                owner, 
                repo_name,
                pr_id,
                diff_hunk,
                reviewer,
                code_review_comment,
                self.dataset, 
                lang
            ))
        return data
