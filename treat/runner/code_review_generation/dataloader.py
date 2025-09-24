import json
import os
from tqdm import tqdm
from .data import CodeReviewData
from typing import List
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_REVIEW_DIR = os.path.join(PROJECT_ROOT, "..", "data", "code_review")

class DataLoader:
    def __init__(self, dataset: str, language: str):
        self.dataset = dataset
        self.language = language
        

    def load_data(self):
        data = []
        with open(os.path.join(CODE_REVIEW_DIR, f"{self.dataset}_cr_dataset.jsonl"), "r") as file:
            lines = file.readlines()
            for line in lines:
                record = json.loads(line)
                lang = record["lang"]
                if lang != self.language:
                    continue
                owner = record["owner"]
                repo_name = record["repo_name"]
                pr_id = record["pr_id"]
                diff_hunk = record["diff_hunk"]
                reviewer = record["reviewer"]
                code_review_comment = record["code_review_comment"]
                data.append(CodeReviewData(owner, repo_name, pr_id, diff_hunk, reviewer, code_review_comment, self.dataset, lang))
        return data
