#!/usr/bin/env python3
"""
TREAT Evaluation Script

Processes raw prediction files using standalone evaluators.
Follows pattern: save/  evaluators / results/<task>/<dataset>/
"""

import os
import sys
import yaml
from glob import glob
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluators.code_review_evaluator import evaluate_code_review_file
from evaluators.code_summarization_evaluator import process_pipeline as evaluate_code_summarization_file
from evaluators.code_translation_evaluator import process_pipeline as evaluate_code_translation_file
from evaluators.code_generation_evaluator import process_pipeline as evaluate_code_generation_file
from evaluators.vulnerability_detection_evaluator import process_pipeline as evaluate_vulnerability_detection_file
from evaluators.symprompt_evaluator import evaluate_symprompt_file

def load_config(config_path: str = "configs/configs.yaml") -> dict:
    """Load evaluation configuration"""
    config_file = project_root / config_path
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config
        return {
            'evaluation': {
                'k_values': [1, 5, 10],
                'num_workers': 4,
                'tasks': ['code_review_generation', 'code_summarization', 'code_translation', 'code_generation', 'vulnerability_detection']
            }
        }


def find_prediction_files(save_dir: str = "save") -> dict:
    """
    Find all prediction files organized by task.
    
    Returns:
        Dict mapping task names to lists of prediction files
    """
    save_path = project_root / save_dir
    files_by_task = {}
    
    if not save_path.exists():
        print(f"Warning: Save directory {save_path} does not exist")
        return files_by_task
    
    # Find all .jsonl files in save directory
    for jsonl_file in save_path.glob("*.jsonl"):
        filename = jsonl_file.name
        
        # Extract task name from filename
        if filename.startswith("code_review_generation_"):
            task = "code_review_generation"
        elif filename.startswith("code_summarization_"):
            task = "code_summarization"
        elif filename.startswith("code_translation_"):
            task = "code_translation"
        elif filename.startswith("code_generation_"):
            task = "code_generation"
        elif filename.startswith("vulnerability_detection_"):
            task = "vulnerability_detection"
        else:
            continue  # Skip unknown file patterns
            
        if task not in files_by_task:
            files_by_task[task] = []
        files_by_task[task].append(str(jsonl_file))
    
    return files_by_task


def evaluate_code_review_files(files: list, config: dict) -> list:
    """Evaluate code review generation files"""
    print(f"\n=== Evaluating Code Review Generation ({len(files)} files) ===")
    
    k_values = config.get('evaluation', {}).get('k_values', [1, 5])
    num_workers = config.get('evaluation', {}).get('num_workers', 4)
    
    results = []
    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            output_file = evaluate_code_review_file(file_path, k_values, num_workers)
            results.append(output_file)
            print(f"  � Saved to: {output_file}")
        except Exception as e:
            print(f"   Error: {e}")
    
    return results


# def evaluate_code_translation_files(files: list, config: dict) -> list:
#     """Evaluate code translation files"""
#     print(f"\n=== Evaluating Code Translation ({len(files)} files) ===")
    
#     k_values = config.get('evaluation', {}).get('k_values', [1, 5])
#     num_workers = config.get('evaluation', {}).get('num_workers', 4)
    
#     results = []
#     for file_path in files:
#         print(f"Processing: {os.path.basename(file_path)}")
#         try:
#             # Use existing code_translation_evaluator.process_pipeline
#             evaluate_code_translation_file(
#                 file_path, 
#                 k_list=k_values,
#                 evaluate_datasets=['hackerrank', 'polyhumaneval'],
#                 num_workers=num_workers
#             )
#             # The code_translation_evaluator handles its own output path
#             print(f"  � Code translation evaluation completed")
#             results.append(file_path)  # Just track that we processed it
#         except Exception as e:
#             print(f"   Error: {e}")
    
#     return results


# def evaluate_code_generation_files(files: list, config: dict) -> list:
#     """Evaluate code generation files"""
#     print(f"\n=== Evaluating Code Generation ({len(files)} files) ===")
    
#     k_values = config.get('evaluation', {}).get('k_values', [1, 5, 10])
#     num_workers = config.get('evaluation', {}).get('num_workers', 4)
    
#     results = []
#     for file_path in files:
#         print(f"Processing: {os.path.basename(file_path)}")
#         try:
#             output_dir = evaluate_code_generation_file(file_path, k_values, num_workers)
#             results.append(output_dir)
#             print(f"  ✅ Saved to: {output_dir}")
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
    
#     return results


def evaluate_files_by_task(task: str, files: list, config: dict) -> list:
    """Evaluate files for a specific task"""
    print(f"\n=== Evaluating {task.replace('_', ' ').title()} ({len(files)} files) ===")
    
    k_values = config.get('evaluation', {}).get('k_values', [1, 5])
    num_workers = config.get('evaluation', {}).get('num_workers', 4)
    
    results = []
    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            if task == "code_review_generation":
                max_k = max(k_values) if k_values else 1
                # Auto-detect file type (predictions vs parsed) and handle accordingly
                output_file = evaluate_code_review_file(file_path, max_k=max_k, judge_model="gpt-4o-2024-11-20", 
                                                       auto_parse=None, extract_only=False, num_workers=num_workers)
            elif task == "code_summarization":
                max_k = max(k_values) if k_values else 1
                output_file = evaluate_code_summarization_file(file_path, max_k=max_k, judge_model="gpt-4o-2024-11-20", 
                                                             auto_parse=None, extract_only=False, num_workers=num_workers)
            elif task == "code_translation":
                output_file = evaluate_code_translation_file(file_path, k_list=k_values, evaluate_datasets=None, num_workers=num_workers)
            elif task == "code_generation":
                output_file = evaluate_code_generation_file(file_path, k_list=k_values, evaluate_datasets=None, num_workers=num_workers)
            elif task == "vulnerability_detection":
                output_file = evaluate_vulnerability_detection_file(file_path, k_list=k_values, evaluate_datasets=None, num_workers=num_workers)
            else:
                print(f"  ⚠️ Unknown task: {task}")
                continue
                
            results.append(output_file)
            print(f"  ✅ Completed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return results


def main():
    """Main evaluation pipeline"""
    print("TREAT Evaluation Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: k_values={config['evaluation']['k_values']}, num_workers={config['evaluation']['num_workers']}")
    
    # Find prediction files by task
    prediction_files = find_prediction_files()
    
    if not prediction_files:
        print("\nNo prediction files found in save/ directory")
        print("Expected filename patterns:")
        print("  - code_review_generation_<model>.jsonl")
        print("  - code_summarization_<model>.jsonl") 
        print("  - code_translation_<model>.jsonl")
        print("  - code_generation_<model>.jsonl")
        print("  - vulnerability_detection_<model>.jsonl")
        return
    
    # Process each task
    all_results = {}
    enabled_tasks = config.get('evaluation', {}).get('tasks', [])
    
    for task, files in prediction_files.items():
        if task not in enabled_tasks:
            print(f"\nSkipping {task} (not enabled in config)")
            continue
            
        task_results = evaluate_files_by_task(task, files, config)
        all_results[task] = task_results
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    total_processed = 0
    for task, results in all_results.items():
        print(f"{task.replace('_', ' ').title()}: {len(results)} files processed")
        total_processed += len(results)
    
    print(f"\nTotal files processed: {total_processed}")
    print(f"Results saved to: results/<task>/<dataset>/evaluations/")

if __name__ == "__main__":
    main()