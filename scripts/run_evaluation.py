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

from file_search_utils.search import find_matching_files
from evaluators.code_review_evaluator import evaluate_code_review_file
from evaluators.code_translation_evaluator import process_pipeline as evaluate_code_translation_file
from evaluators.code_generation_evaluator import process_pipeline as evaluate_code_generation_file

# from evaluators.code_generation_evaluator import evaluate_code_generation_file
from evaluators.vulnerability_detection_evaluator import process_pipeline as evaluate_vulnerability_detection_file
from extractors.llm_extraction_utils.code_translation_extractor import CodeTranslationExtractor
from extractors.llm_extraction_utils.code_generation_extractor import CodeGenerationExtractor
from extractors.llm_extraction_utils.symprompt_extractor import SymPromptExtractor
from evaluators.symprompt_evaluator import SympromptEvaluator

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
                'tasks': ['code_review_generation', 'code_translation', 'code_generation']
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
        elif filename.startswith("code_translation_"):
            task = "code_translation"
        elif filename.startswith("code_generation_"):
            task = "code_generation"
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


def main():
    """Main evaluation pipeline"""
    print("TREAT Evaluation Pipeline")

    ######### TODO: CODE TRANSLATION
    # files = find_matching_files("results/code_translation/polyhumaneval/predictions", None, ".jsonl")
    # print(files)
    # files = find_matching_files("results/code_translation/hackerrank/predictions", None, ".jsonl")
    # files = find_matching_files("results/code_translation/polyhumaneval/parsed", None, ".jsonl")
    # print(files)
    # CodeTranslationExtractor("gpt-5-nano").extract("results/code_translation/hackerrank/predictions/Grok-3-Mini (High).jsonl")
    
    # for file in files[:]:
    #     for model in ('Claude-3.5', 'Claude-3.7', 'DeepSeek-R1', 'GPT-3.5', 'GPT-4', 'Grok-3-Mini', 'Llama-3.1-70B', 'Llama-3.3-70B', 'Llama-4', 'Qwen3', 'Qwen2.5-7B', 'Gemma', 'Gemini', 'o4'):
    #         if model in file: 
    #             continue
    
        # CodeTranslationExtractor("gpt-5-nano").extract(file)
    # files = ["/Users/ericjohnli/Downloads/TREAT-refined/results/code_translation/polyhumaneval/parsed/GPT-5.jsonl"]/
    # evaluate_code_translation_file(files[0], k_list=[1], evaluate_datasets=['polyhumaneval'], num_workers=10)

    # files = find_matching_files("results/code_translation/polyhumaneval/parsed", None, ".jsonl")
    # # count = 0
    # for file in files:
    #     in_file = False
    #     if 'GPT-5' in file:
    #         continue
    #     # for model in ('Claude-3.5', 'Claude-3.7', 'DeepSeek-R1', 'GPT-3.5', 'GPT-4', 'Grok-3-Mini', 'Llama-3.1-70B', 'Llama-3.3-70B', 'Llama-4', 'Qwen3', 'Qwen2.5-7B', 'Gemma', 'Gemini', 'o4', 'Grok'):
    #         # if model in file: 
    #         #     # count += 1
    #         #     print(model)
    #         #     in_file = True
    #         #     # print("SKIP")
    #         #     break
    #     # if in_file:
    #     #     print("SKIP")
    #     #     continue
    # # print(count)
    # file = "/Users/ericjohnli/Downloads/TREAT-refined/results/code_translation/hackerrank/parsed/Grok-3-Mini (High).jsonl"
    # evaluate_code_translation_file(file, k_list=[1], evaluate_datasets=['hackerrank'], num_workers=8)
    
    ########## TODO: CODE GENERATION
    # file = '/Users/ericjohnli/Downloads/TREAT-refined/results/code_generation/hackerrank/predictions/GPT-5.jsonl'
    # file = CodeGenerationExtractor('gpt-5-nano').extract(file)
    files = find_matching_files("results/code_generation/geeksforgeeks/predictions", None, ".jsonl")

    # # files = find_matching_files("results/code_generation/hackerrank/parsed", None, ".jsonl")
    # files = [
    #     # "results/code_generation/hackerrank/parsed/Llama-4-Scout-17B-16E-Instruct.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Qwen3-235B-A22B.jsonl",
    #     # "results/code_generation/hackerrank/parsed/DeepSeek-R1.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Qwen3-32B.jsonl",
    #     # "results/code_generation/hackerrank/parsed/GPT-4-turbo-2024-04-09.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Llama-3.3-70B-Instruct.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Claude-3.5-Haiku-20241022.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Qwen3-30B-A3B.jsonl",
    #     # "results/code_generation/hackerrank/parsed/GPT-5.jsonl",
    #     # "results/code_generation/hackerrank/parsed/DeepSeek-V3 (0324).jsonl",
    #     # "results/code_generation/hackerrank/parsed/o3-mini (Med).jsonl",
    #     # "results/code_generation/hackerrank/parsed/Claude-Sonnet-4.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Llama-3.1-8B-Instruct.jsonl",
    #     # "results/code_generation/hackerrank/parsed/Qwen2.5-Coder-32B-Instruct.jsonl",
    # ]
    # file = "/Users/ericjohnli/Downloads/TREAT-refined/results/code_generation/hackerrank/parsed/GPT-5.jsonl"
    # for file in files:
        # if 'gpt-5' in file or 'Gemini' in file or 'Grok' in file:
            # print("SKIP")
            # continue
        # print(file)
        # file = "/Users/ericjohnli/Downloads/TREAT-refined/results/code_generation/geeksforgeeks/parsed/GPT-5.jsonl"    
    # file = CodeGenerationExtractor('gpt-5-nano').extract(file)
        
    files = find_matching_files("results/code_generation/geeksforgeeks/evaluations", None, ".jsonl")
    file = "results/code_review_generation/github_2023/parsed/Llama-4-Scout-17B-16E-Instruct.jsonl"
    evaluate_code_review_file(file, 1, )
    # # # Example: run a single code generation file
    # # # file = '/Users/ericjohnli/Downloads/TREAT-refined/results/code_generation/geeksforgeeks/parsed/o3-mini (Med).jsonl'
    start=False 
    from pprint import pprint
    print(files)
    # for file in files[3:]:
        # if 'Haiku' not in file and 'Sonnet-4' not in file and 'GPT-3.5' not in file \
        #     and 'GPT-4.1' not in file and 'GPT-4o' not in file and 'Llama-3.1-8B' not in file\
        #     and 'Qwen3-30B' not in file:
        #     continue

        # if 'Claude-3.5-Haiku' not in file and not start:
        #     print("SKIP")
        #     continue
        # if 'Claude-3.5-Haiku' in file:
        #     start=True
        # if 'GPT-5' in file:
        #     print("SKIP")
        #     continue
        # if 'Grok-3' in file:
        #     print("SKIP")
        #     continue
        # if 'DeepSeek-R1' in file:
        #     print("SKIP")
        #     continue
        # if '3.5-Sonnet' in file:
        #     print("SKIP")
        #     continue
        # if 'o4' in file:
        #     print("SKIP")
        #     continue
        # if 'Gemini' in file:
        #     print("SKIP")
        #     continue
        # if 'Gemma' in file:
        #     print("SKIP")
        #     continue
        # if '3.7' in file:
        #     print("SKIP")
        #     continue
        # if 'Llama-4' in file:
        #     print("SKIP")
        #     continue
        # if '235B' in file:
        #     print("SKIP")
        #     continue
        # if 'Qwen3-32B' in file:
        #     print("SKIP")
        #     continue
        # if 'GPT-4-turbo' in file:
        #     print("SKIP")
        #     continue
        # if 'Llama-3.3' in file:
        #     print("SKIP")
        #     continue
        # if 'DpeeSee'
        
        # pass
        # if 'GPT' in file:
        #     continue
    # files = ["results/code_generation/geeksforgeeks/parsed/GPT-5.jsonl"]
    # # file = files[0]
        # evaluate_code_generation_file(file, k_list=[1], evaluate_datasets=['geeksforgeeks'], num_workers=8)
    # file = "results/code_translation/hackerrank/evaluations/DeepSeek-R1.jsonl"
    # evaluate_code_translation_file(file, k_list=[1], evaluate_datasets=['hackerrank'], num_workers=8)
    # Symprompt: evaluate parsed JSONL files with the Symprompt evaluator
    # from evaluators.symprompt_evaluator import evaluate_symprompt_file as evaluate_file

    # symprompt_parsed = find_matching_files("results/unit_test_generation/symprompt/parsed", None, ".jsonl")
    # if symprompt_parsed:
    #     print(f"\n=== Evaluating Symprompt unit_test_generation ({len(symprompt_parsed)} files) ===")
    #     evaluator = SympromptEvaluator()
    #     for fp in symprompt_parsed[26:]:
    #         print(f"Processing: {os.path.basename(fp)}")
    #         if 'GPT-5' in fp or 'Grok' in fp or 'DeepSeek-R1 (0528)' in fp or 'Claude-3.5-Sonnet' in fp or 'Gemma-3' in fp or 'GPT-4o' in fp or 'o4' in fp or 'Qwen2.5-72B' in fp or 'Llama-3.1-70B' in fp or 'GPT-4.1' in fp or 'GPT-3.5' in fp or 'Claude-3.7' in fp or 'Llama-4' in fp:
    #             print("SKIP")
    #             continue
    #         try:
    #             stats = evaluate_file(fp)
    #             print(f"  ✅ Completed: {os.path.basename(fp)}")
    #             # print(f"  Stats: total={stats.get('total_cases')}, exec={stats.get('executable_cases')}, avg_line={stats.get('coverage_stats',{}).get('avg_line_coverage')}")
    #         except Exception as e:
    #             print(f"  ❌ Error: {e}")
    # file = '/Users/ericjohnli/Downloads/TREAT-refined/results/unit_test_generation/symprompt/parsed/GPT-5.jsonl'
    # evaluator = SympromptEvaluator()
    # from evaluators.symprompt_evaluator import evaluate_symprompt_file
    # file ='/Users/ericjohnli/Downloads/TREAT-refined/results/unit_test_generation/symprompt/parsed/Gemini-2.5-Pro-Preview-05-06.jsonl'
    # evaluate_symprompt_file(file, num_workers=4, run_error_items=False)
    # file = "/Users/ericjohnli/Downloads/TREAT-refined/results/code_translation/polyhumaneval/evaluations/Claude-Sonnet-4.jsonl"
    # files = find_matching_files("/Users/ericjohnli/Downloads/TREAT-refined/results/code_translation/polyhumaneval/evaluations", None, ".jsonl")
    # print(files)
    # start = False
    # for file in files:
    #     if 'Llama-3.1-8B' not in file and not start:
    #         continue
    #     if 'Llama-3.1-8B' in file:
    #         start = True
    #     if start:
            # evaluate_code_translation_file(file, k_list=[1], evaluate_datasets=['polyhumaneval'], num_workers=8)
    ######### TODO: VULNERABILITY DETECTION
    # files = find_matching_files("results/vulnerability_detection/primevul/predictions/", None, ".jsonl")
    # print(files)
    # files = ["results/vulnerability_d"]
    # evaluate_vulnerability_detection_file(files)


    


if __name__ == "__main__":
    main()