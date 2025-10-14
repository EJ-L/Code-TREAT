#!/usr/bin/env python
"""
Script to download and set up all required datasets for TREAT.
This script downloads datasets from Hugging Face and sets them up in the correct locations.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datasets import load_dataset
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define project root
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

# Define dataset mappings
DATASET_MAPPINGS = {
    "code_translation": {
        "lite": "Code-TREAT/code_translation_lite",
        "full": "Code-TREAT/code_translation"
    },
    "code_generation": {
        "lite": "Code-TREAT/code_generation_lite",
        "full": "Code-TREAT/code_generation"
    },
    "code_summarization": {
        "lite": "Code-TREAT/code_summarization_lite",
        "full": "Code-TREAT/code_summarization"
    },
    "code_reasoning": {
        "lite": "Code-TREAT/code_reasoning_lite",
        "full": "Code-TREAT/code_reasoning"
    },
    "code_review": {
        "lite": "Code-TREAT/code_review_lite",
        "full": "Code-TREAT/code_review"
    },
    "unit_test_generation": {
        "lite": "Code-TREAT/unit_test_generation",
        "full": "Code-TREAT/unit_test_generation"
    },
    "vulnerability_detection": {
        "primevul_lite": "Code-TREAT/PrimeVul_original_lite",
        "primevul_full": "Code-TREAT/PrimeVul_original",
        "primevul_pair_lite": "Code-TREAT/PrimeVul-Paired_original_lite",
        "primevul_pair_full": "Code-TREAT/PrimeVul-Paired_original"
    }
}

def ensure_dir_exists(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

def download_dataset(dataset_name, output_dir, split="test"):
    """Download a dataset from Hugging Face and save it locally."""
    logging.info(f"Downloading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
        
        # Create output directory
        ensure_dir_exists(output_dir)
        
        # Save dataset to JSON
        output_file = os.path.join(output_dir, f"{dataset_name.split('/')[-1]}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(dataset[split]), f, ensure_ascii=False, indent=2)
        
        logging.info(f"Dataset saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error downloading dataset {dataset_name}: {e}")
        return None

def setup_polyhumaneval_dataset():
    """Set up the PolyHumanEval dataset."""
    logging.info("Setting up PolyHumanEval dataset")
    
    # Create directory for PolyHumanEval data
    poly_dir = ensure_dir_exists(DATA_DIR / "polyhumaneval")
    
    # Download the dataset
    dataset_name = DATASET_MAPPINGS["code_translation"]["lite"]
    try:
        dataset = load_dataset(dataset_name)
        
        # Extract PolyHumanEval data
        poly_data = {}
        for lang in ["python", "java", "cpp", "go", "js"]:
            poly_data[lang] = {}
        
        for item in dataset["test"]:
            if "polyhumaneval" in item.get("domain", "").lower():
                lang = item["source_language"]
                if lang in poly_data:
                    poly_data[lang][item['id']] = item['source_code']
        
        # Save PolyHumanEval data
        poly_file = poly_dir / "polyhumaneval_sol.json"
        with open(poly_file, 'w', encoding='utf-8') as f:
            json.dump(poly_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"PolyHumanEval data saved to {poly_file}")
        return poly_file
    except Exception as e:
        logging.error(f"Error setting up PolyHumanEval dataset: {e}")
        return None

def setup_all_datasets(use_lite=True):
    """Set up all datasets."""
    # Create data directory
    ensure_dir_exists(DATA_DIR)
    
    # Set up PolyHumanEval dataset
    setup_polyhumaneval_dataset()
    
    # Download other datasets
    for task, datasets in DATASET_MAPPINGS.items():
        task_dir = ensure_dir_exists(DATA_DIR / task)
        
        if task == "vulnerability_detection":
            # Special handling for vulnerability detection
            primevul_dir = ensure_dir_exists(task_dir / "primevul")
            primevul_pair_dir = ensure_dir_exists(task_dir / "primevul_pair")
            
            if use_lite:
                download_dataset(datasets["primevul_lite"], primevul_dir)
                download_dataset(datasets["primevul_pair_lite"], primevul_pair_dir)
            else:
                download_dataset(datasets["primevul_full"], primevul_dir)
                download_dataset(datasets["primevul_pair_full"], primevul_pair_dir)
        else:
            # Standard handling for other tasks
            dataset_name = datasets["lite"] if use_lite else datasets["full"]
            download_dataset(dataset_name, task_dir)

def main():
    """Main function."""
    print("TREAT Dataset Setup")
    print("==================")
    print("This script will download and set up all required datasets for TREAT.")
    print("Datasets will be downloaded from Hugging Face and saved locally.")
    print()
    
    use_lite = input("Use lite versions of datasets? (y/n) [y]: ").lower() != 'n'
    
    print()
    print(f"Using {'lite' if use_lite else 'full'} versions of datasets.")
    print("Starting download...")
    print()
    
    setup_all_datasets(use_lite)
    
    print()
    print("Dataset setup complete!")
    print("You can now run experiments with the downloaded datasets.")

if __name__ == "__main__":
    main()
