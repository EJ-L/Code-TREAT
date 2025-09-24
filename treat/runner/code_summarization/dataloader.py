from pathlib import Path
import json
from .data import CodeSumData

class DataLoader:
    def __init__(self, dataset: str, language: str,):
        self.dataset = dataset
        self.language = language

    def _get_file_path(self) -> Path:
        """Generate full file path for the language-specific data file."""
        dataset_dirs = {
            "github_2023": Path("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_summarization/parsed_functions"),
        }
        dataset_filename = {
            "github_2023": f"parsed_{self.language}_functions.jsonl",
        }
        return dataset_dirs[self.dataset] / dataset_filename[self.dataset]
    
    def load_data(self):
        file_path = self._get_file_path()
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        data = []
        with open(file_path) as reader:
            for line in reader:
                data.append(CodeSumData(json.loads(line)))
        return data
