#!/usr/bin/env python3

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def get_model_name_from_filename(filename):
    """Extract model name from evaluation filename by removing task prefix and .jsonl suffix"""
    if filename.startswith('vulnerability_detection_'):
        return filename.replace('vulnerability_detection_', '').replace('.jsonl', '')
    
    # For other tasks, just remove .jsonl
    return filename.replace('.jsonl', '')


def filter_code_summarization_fields(data):
    """Filter code_summarization data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'lang', 'metrics']
    
    if isinstance(data, dict):
        return {k: data[k] for k in field_order if k in data}
    else:
        return data


def filter_code_translation_fields(data):
    """Filter code_translation data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'source_lang', 'target_lang', 'modality', 'model_name', 'difficulty', 'metrics']
    
    if isinstance(data, dict):
        filtered_data = {k: data[k] for k in field_order if k in data}
        # Add default difficulty if missing
        if 'difficulty' not in filtered_data:
            filtered_data['difficulty'] = 'Easy'
        return filtered_data
    else:
        return data


def filter_code_generation_fields(data):
    """Filter code_generation data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'lang', 'model_name', 'difficulty', 'metrics']
    
    if isinstance(data, dict):
        filtered_data = {k: data[k] for k in field_order if k in data}
        # Add default difficulty if missing
        if 'difficulty' not in filtered_data:
            filtered_data['difficulty'] = 'Easy'
        return filtered_data
    else:
        return data


def filter_code_review_generation_fields(data):
    """Filter code_review_generation data to keep only specified fields"""
    field_order = ['task', 'dataset', 'lang', 'model_name', 'metrics']
    
    if isinstance(data, dict):
        return {k: data[k] for k in field_order if k in data}
    else:
        return data


def filter_input_prediction_fields(data):
    """Filter input_prediction data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'lang', 'model_name', 'difficulty', 'metrics']
    
    if isinstance(data, dict):
        filtered_data = {k: data[k] for k in field_order if k in data}
        # Add default difficulty if missing
        if 'difficulty' not in filtered_data:
            filtered_data['difficulty'] = 'Easy'
        return filtered_data
    else:
        return data


def filter_output_prediction_fields(data):
    """Filter output_prediction data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'lang', 'model_name', 'difficulty', 'metrics']
    
    if isinstance(data, dict):
        filtered_data = {k: data[k] for k in field_order if k in data}
        # Add default difficulty if missing
        if 'difficulty' not in filtered_data:
            filtered_data['difficulty'] = 'Easy'
        return filtered_data
    else:
        return data


def filter_unit_test_generation_fields(data):
    """Filter unit_test_generation data to keep only specified fields"""
    field_order = ['task', 'dataset', 'ref_key', 'lang', 'metrics']
    
    if isinstance(data, dict):
        return {k: data[k] for k in field_order if k in data}
    else:
        return data


def merge_jsonl_files(file_paths, output_path, task_type):
    """Merge multiple JSONL files into one, applying filtering if needed"""
    merged_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Apply filtering for specific tasks
                        if task_type == 'code_summarization':
                            data = filter_code_summarization_fields(data)
                        elif task_type == 'code_translation':
                            data = filter_code_translation_fields(data)
                        elif task_type == 'code_generation':
                            data = filter_code_generation_fields(data)
                        elif task_type == 'code_review_generation':
                            data = filter_code_review_generation_fields(data)
                        elif task_type == 'input_prediction':
                            data = filter_input_prediction_fields(data)
                        elif task_type == 'output_prediction':
                            data = filter_output_prediction_fields(data)
                        elif task_type == 'unit_test_generation':
                            data = filter_unit_test_generation_fields(data)
                        
                        merged_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {file_path}: {e}")
                        continue
    
    # Write merged data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Merged {len(merged_data)} entries to {output_path}")


def process_code_translation_task(results_dir, dest_dir, datasets):
    """Process code_translation task with special field filtering"""
    print("Processing task: code_translation (with field filtering)")
    
    # Group models by name for this task
    model_files = defaultdict(list)
    
    for dataset in datasets:
        evaluations_dir = os.path.join(results_dir, 'code_translation', dataset, 'evaluations')
        
        if not os.path.exists(evaluations_dir):
            print(f"Warning: Evaluations directory not found: {evaluations_dir}")
            continue
        
        # List all .jsonl files in evaluations directory
        for filename in os.listdir(evaluations_dir):
            if filename.endswith('.jsonl'):
                model_name = get_model_name_from_filename(filename)
                file_path = os.path.join(evaluations_dir, filename)
                model_files[model_name].append(file_path)
    
    # Merge files for each model
    for model_name, file_paths in model_files.items():
        if len(file_paths) > 1:
            print(f"  Merging {len(file_paths)} files for model: {model_name}")
        
        # Create output path
        task_dest_dir = os.path.join(dest_dir, 'code_translation')
        output_path = os.path.join(task_dest_dir, f"{model_name}.jsonl")
        
        merge_jsonl_files(file_paths, output_path, 'code_translation')


def process_task(results_dir, dest_dir, task, datasets):
    """Process a specific task and its datasets"""
    print(f"Processing task: {task}")
    
    # Group models by name for this task
    model_files = defaultdict(list)
    
    for dataset in datasets:
        evaluations_dir = os.path.join(results_dir, task, dataset, 'evaluations')
        
        if not os.path.exists(evaluations_dir):
            print(f"Warning: Evaluations directory not found: {evaluations_dir}")
            continue
        
        # List all .jsonl files in evaluations directory
        for filename in os.listdir(evaluations_dir):
            if filename.endswith('.jsonl'):
                model_name = get_model_name_from_filename(filename)
                file_path = os.path.join(evaluations_dir, filename)
                model_files[model_name].append(file_path)
    
    # Merge files for each model
    for model_name, file_paths in model_files.items():
        if len(file_paths) > 1:
            print(f"  Merging {len(file_paths)} files for model: {model_name}")
        
        # Create output path
        task_dest_dir = os.path.join(dest_dir, task)
        output_path = os.path.join(task_dest_dir, f"{model_name}.jsonl")
        
        merge_jsonl_files(file_paths, output_path, task)


def handle_vulnerability_detection(results_dir, dest_dir):
    """Handle vulnerability_detection special case"""
    print("Processing vulnerability_detection (special case)")
    
    vul_score_path = os.path.join(results_dir, 'vulnerability_detection', 'primevul', 'evaluations', 'vul_detect_score.json')
    
    if os.path.exists(vul_score_path):
        dest_vul_dir = os.path.join(dest_dir, 'vulnerability-detection')
        os.makedirs(dest_vul_dir, exist_ok=True)
        
        dest_score_path = os.path.join(dest_vul_dir, 'vul_detect_score.json')
        shutil.copy2(vul_score_path, dest_score_path)
        print(f"Copied vul_detect_score.json to {dest_score_path}")
    else:
        print(f"Warning: vul_detect_score.json not found at {vul_score_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate codetreat results by merging evaluation files')
    parser.add_argument('dest', help='Destination directory for output')
    parser.add_argument('--results-dir', default='results', help='Source results directory (default: results)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    results_dir = os.path.abspath(args.results_dir)
    dest_dir = os.path.abspath(args.dest)
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"Source directory: {results_dir}")
    print(f"Destination directory: {dest_dir}")
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define task mappings (task_name -> list of datasets)
    task_mappings = {
        'code_generation': ['geeksforgeeks', 'hackerrank'],
        'code_review_generation': ['github_2023'],
        'code_summarization': ['github_2023'],
        'code_translation': ['hackerrank', 'polyhumaneval'],
        'unit_test_generation': ['llm4ut', 'symprompt'],
        'input_prediction': ['geeksforgeeks', 'hackerrank'],
        'output_prediction': ['geeksforgeeks', 'hackerrank']
    }
    
    # Process each task
    for task, datasets in task_mappings.items():
        if task == 'vulnerability_detection':
            continue  # Handle separately
        elif task == 'code_translation':
            # Special handling for code_translation - single merged file
            task_dir = os.path.join(results_dir, task)
            if os.path.exists(task_dir):
                process_code_translation_task(results_dir, dest_dir, datasets)
            else:
                print(f"Warning: Task directory not found: {task_dir}")
        else:
            task_dir = os.path.join(results_dir, task)
            if os.path.exists(task_dir):
                process_task(results_dir, dest_dir, task, datasets)
            else:
                print(f"Warning: Task directory not found: {task_dir}")
    
    # Handle vulnerability_detection special case
    handle_vulnerability_detection(results_dir, dest_dir)
    
    print("Processing complete!")


if __name__ == '__main__':
    main()