"""
Code Translation Evaluator

Standalone evaluator that processes raw prediction files from code translation tasks
and adds evaluation metrics. Follows the pattern: save/ → evaluators → results/

Supports:
- HackerRank benchmark evaluation with test case execution
- geeksforgeeks benchmark evaluation with project template system
- Pass@k evaluation metrics
- Parallel processing for improved performance
"""

# Standard library
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# Shared evaluation utilities
from evaluators.utils import (
    CodeExecutor,
    decompress_test_cases,
    compute_pass_at_k_from_results,
    get_optimal_worker_count,
)

# Tree-sitter utilities for code formatting
from extractors.tree_sitter_extraction_utils.python_tree_sitter_utils import orchestrate_code_extraction_filtered as remove_python_example_usage, PYTHON_BASE_IMPORTS, python_put_under_class
from extractors.tree_sitter_extraction_utils.java_tree_sitter_utils import JAVA_BASE_IMPORTS, clean_hackerrank_java_code, clean_geeksforgeeks_java_code

### imports that may be needed for python

# Global executor instance
EXECUTOR = CodeExecutor()

# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "code_generation")

# Dataset-specific paths
HACKERRANK_TESTCASE_PATH = os.path.join(
    PROJECT_ROOT, "data", "hackerrank", "hackerrank_filtered.jsonl"
)
GEEKSFORGEEKS_TESTCASE_PATH = os.path.join(
    PROJECT_ROOT, "data", "geeksforgeeks", "geeksforgeeks_filtered_valid.jsonl"
)

def get_hackerrank_testcases() -> Dict[int, List[Dict[str, str]]]:
    testcase_hash = {}
    driver_hash = {}
    java_func_sign_info = {}
    with open(HACKERRANK_TESTCASE_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            _id = data['question_id']
            testcase_hash[_id] = decompress_test_cases(data['test_cases'])
            if _id not in driver_hash:
                driver_hash[_id] = {}
            for lang in ['python', 'java']:
                driver_hash[_id][lang] = (data[lang]['template_head'], data[lang]['template_tail'])
            java_func_sign_info[_id] = (data['java']['class_name'], data['java']['func_sign'], data['java']['script_name'])
                
    return testcase_hash, driver_hash, java_func_sign_info

def get_geeksforgeeks_testcases() -> Dict[int, List[Dict[str, str]]]:
    testcase_hash = {}
    driver_hash = {}
    func_sign_info = {}
    user_hash = {}
    with open(GEEKSFORGEEKS_TESTCASE_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            _id = data['question_id']
            testcase_hash[_id] = decompress_test_cases(data['test_cases'])
            if _id not in driver_hash:
                driver_hash[_id] = {} 
            if _id not in func_sign_info:
                func_sign_info[_id] = {} 
            if _id not in user_hash:
                user_hash[_id] = {}
            for lang in ['python', 'java']:
                driver_hash[_id][lang] = data[lang]['initial_code']
                user_hash[_id][lang] = data[lang]['user_code']
                func_sign_info[_id][lang] = (data[lang]['class_name'], data[lang]['func_sign'], data[lang].get("script_name", ""))
    return testcase_hash, driver_hash, user_hash, func_sign_info

def _process_hackerrank_item(args):
    """Process a single HackerRank item - thread-safe version."""
    data, k_list, testcase_hash, driver_hash, java_func_sign_info = args
    
    try:
        _id = data['id']
        language = data['lang']
        predictions = data['parsed_response'][:min(max(k_list), len(data['parsed_response']))]
        
        # Check if required data exists
        if _id not in testcase_hash:
            data['executed_code'] = []
            data['results'] = [{'error': 'no test cases'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
            
        if _id not in driver_hash or language not in driver_hash[_id]:
            data['executed_code'] = []
            data['results'] = [{'error': f'no driver code for {language}'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
        
        driver_code = driver_hash[_id][language]
        prompt_id = data['prompting_category'] + "_" + str(data['prompt_id'])
        # Create fresh executors for this thread to avoid contention
        test_cases = testcase_hash[_id]
        test_inputs = [test_case['input'] for test_case in test_cases]
        test_outputs = [test_case['expected_output'] for test_case in test_cases]
        # direct_3 should be a directly executable java/python code
        if prompt_id != "direct_3":
            predictions = [
                clean_hackerrank_java_code(pred, driver_code, java_func_sign_info[_id]) if language == 'java'  # add necessary imports
                else PYTHON_BASE_IMPORTS + '\n' + remove_python_example_usage(pred) + '\n' + driver_code[1]
                for pred in predictions
            ]
        else:
            predictions = [
                JAVA_BASE_IMPORTS + '\n' + pred if language == 'java' 
                else PYTHON_BASE_IMPORTS + '\n' + pred
                for pred in predictions
            ]
        
        results = []
        # Process predictions for this item
        for _, code in enumerate(predictions):
            res = EXECUTOR.run_code(language, code, test_inputs, test_outputs)                    
            results.append(res)
        data['executed_code'] = predictions
        data['results'] = results
        data['metrics'] = compute_pass_at_k_from_results(results, k_list)
        return data
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error processing HackerRank item {language} {_id}: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        data['executed_code'] = []
        data['results'] = [{'error': error_msg}]
        data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
        return data
def _get_pass1(metrics: Any) -> float:
    """尽量稳健地取出 pass@1（支持多种命名），取不到则返回 None。"""
    if not isinstance(metrics, dict):
        return None
    for k in ("pass@1", "pass_at_1", "pass1"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                return None
    return None

def _process_geeksforgeeks_item(args):
    """Process a single GeeksforGeeks item - thread-safe version."""
    data, k_list, testcase_hash, driver_hash, user_hash, func_sign_info = args
    
    try:
        _id = data['id']
        language = data['lang']
        predictions = data['parsed_response'][:min(max(k_list), len(data['parsed_response']))]
        prompt_id = data['prompting_category'] + "_" + str(data['prompt_id'])
        # if prompt_id != "direct_3" and data['metrics']['pass@1'] != 0:
        #     return data
        # Check if required data exists
        if _id not in testcase_hash:
            data['executed_code'] = []
            data['results'] = [{'error': 'no test cases'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
            
        if _id not in driver_hash or language not in driver_hash[_id]:
            data['executed_code'] = []
            data['results'] = [{'error': f'ID {_id} not found in dataset - no driver code for {language}'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
            
        if _id not in user_hash or language not in user_hash[_id]:
            data['executed_code'] = []
            data['results'] = [{'error': f'ID {_id} not found in dataset - no user code for {language}'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
        
        driver_code = driver_hash[_id][language]
        user_code = user_hash[_id][language]

        
        
        # Create fresh executors for this thread to avoid contention
        test_cases = testcase_hash[_id]
        test_inputs = [test_case['input'] for test_case in test_cases]
        test_outputs = [test_case['expected_output'] for test_case in test_cases]
        # direct_3 should be a directly executable java/python code
        if prompt_id != "direct_3":
            predictions = [
                clean_geeksforgeeks_java_code(pred, driver_code, func_sign_info[_id]['java']) if language == 'java'  # add necessary imports
                else PYTHON_BASE_IMPORTS + '\n' + python_put_under_class(remove_python_example_usage(pred), user_code, func_sign_info[_id]['python']) + '\n' + driver_code # this allows the case that models forgot to add the main_if 
                for pred in predictions
            ]
        else:
            predictions = [
                JAVA_BASE_IMPORTS + '\n' + pred if language == 'java' 
                else PYTHON_BASE_IMPORTS + '\n' + pred
                for pred in predictions
            ]
        
        results = []
        # Process predictions for this item
        for _, code in enumerate(predictions):
            res = EXECUTOR.run_code(language, code, test_inputs, test_outputs)                    
            results.append(res)
        data['executed_code'] = predictions
        data['results'] = results
        data['metrics'] = compute_pass_at_k_from_results(results, k_list)
        return data
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error processing GeeksforGeeks item {_id} {language}: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        data['executed_code'] = []
        data['results'] = [{'error': error_msg}]
        data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
        return data

def run_hackerrank(hackerrank_data: Dict[str, Any], k_list: List[int], num_workers: int = None) -> Dict[str, Any]:
    """Improved parallel HackerRank evaluation with item-level parallelization."""
    print(f"Starting parallel HackerRank evaluation with {len(hackerrank_data)} items")
    
    # Get test cases once (shared across all workers)
    testcase_hash, driver_hash, java_func_sign_info  = get_hackerrank_testcases()
    
    # Determine optimal worker count - limit to avoid overwhelming system
    if num_workers is None:
        num_workers = min(8, get_optimal_worker_count(len(hackerrank_data)))
    else:
        num_workers = min(num_workers, 8)  # Cap at 4 to prevent resource issues
    
    print(f"Using {num_workers} workers for HackerRank evaluation")
    
    # Prepare arguments for each item
    item_args = [(data, k_list, testcase_hash, driver_hash, java_func_sign_info) for data in hackerrank_data]
    
    valid_hackerrank_data = []
    # Process items in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(
            executor.map(_process_hackerrank_item, item_args),
            total=len(item_args),
            desc="Processing HackerRank items"
        ))
        
        # Filter out None results (failed items)
        valid_hackerrank_data = [result for result in results if result is not None]
    
    print(f"Completed HackerRank evaluation: {len(valid_hackerrank_data)} valid items")
    return valid_hackerrank_data


def run_geeksforgeeks(geeksforgeeks_data: Dict[str, Any], k_list: List[int], num_workers: int = None) -> Dict[str, Any]:
    """Improved parallel GeeksforGeeks evaluation with item-level parallelization."""
    print(f"Starting parallel GeeksforGeeks evaluation with {len(geeksforgeeks_data)} items")
    
    # Get test cases once (shared across all workers)
    testcase_hash, driver_hash, user_hash, func_sign_info  = get_geeksforgeeks_testcases()
    
    # Determine optimal worker count - limit to avoid overwhelming system
    if num_workers is None:
        num_workers = min(8, get_optimal_worker_count(len(geeksforgeeks_data)))
    else:
        num_workers = min(num_workers, 15)  # Cap at 4 to prevent resource issues
    
    print(f"Using {num_workers} workers for GeeksforGeeks evaluation")
    
    # Prepare arguments for each item
    item_args = [(data, k_list, testcase_hash, driver_hash, user_hash, func_sign_info) for data in geeksforgeeks_data]
    
    valid_geeksforgeeks_data = []
    # Process items in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(
            executor.map(_process_geeksforgeeks_item, item_args),
            total=len(item_args),
            desc="Processing GeeksforGeeks items"
        ))
        
        # Filter out None results (failed items)
        valid_geeksforgeeks_data = [result for result in results if result is not None]
    
    print(f"Completed GeeksforGeeks evaluation: {len(valid_geeksforgeeks_data)} valid items")
    return valid_geeksforgeeks_data


def evaluate_hackerrank(filename: str, k_list: List[int], num_workers: int = None) -> str:
    hackerrank_data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            data = json.loads(line)
            hackerrank_data.append(data)
    
    evaluated_data = run_hackerrank(hackerrank_data, k_list, num_workers)

    # Get the directory of the input file, go back one level, then into 'evaluations' directory
    file_dir = os.path.dirname(filename)
    parent_dir = os.path.dirname(file_dir)
    evaluation_dir = os.path.join(parent_dir, 'evaluations')
        
    # Create evaluation directory if it doesn't exist
    os.makedirs(evaluation_dir, exist_ok=True)
    # Create output filename in the evaluation directory
    base_filename = os.path.basename(filename)
    evaluation_file = os.path.join(evaluation_dir, base_filename)

    with open(evaluation_file, "w", encoding="utf-8") as f:
        for d in evaluated_data:
            f.write(json.dumps(d) + '\n')
    
    return evaluation_file

def evaluate_geeksforgeeks(filename: str, k_list: List[int], num_workers: int = None) -> str:
    hackerrank_data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            data = json.loads(line)
            hackerrank_data.append(data)
    
    evaluated_data = run_geeksforgeeks(hackerrank_data, k_list, num_workers)

    # Get the directory of the input file, go back one level, then into 'evaluations' directory
    file_dir = os.path.dirname(filename)
    parent_dir = os.path.dirname(file_dir)
    evaluation_dir = os.path.join(parent_dir, 'evaluations')
        
    # Create evaluation directory if it doesn't exist
    os.makedirs(evaluation_dir, exist_ok=True)
    # Create output filename in the evaluation directory
    base_filename = os.path.basename(filename)
    evaluation_file = os.path.join(evaluation_dir, base_filename)

    with open(evaluation_file, "w", encoding="utf-8") as f:
        for d in evaluated_data:
            f.write(json.dumps(d) + '\n')
    
    return evaluation_file

def process_pipeline(filename: str, k_list: List[int], evaluate_datasets: List[str] = None, num_workers: int = None) -> None:
    print(f"Processing {filename}")
    # Default to evaluate all datasets if none specified
    if evaluate_datasets is None:
        evaluate_datasets = ['hackerrank', 'polyhumaneval']
    print(evaluate_datasets)
    
    try:
        # Run evaluations sequentially to avoid resource contention
        print("Running evaluations sequentially (each dataset internally parallel)...")
        if 'hackerrank' in evaluate_datasets:
            # Run HackerRank first
            evaluate_hackerrank(filename, k_list, num_workers)
        if 'geeksforgeeks' in evaluate_datasets:
            # Run geeksforgeeks second
            evaluate_geeksforgeeks(filename, k_list, num_workers)
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        # Don't remove files if there was an error
        raise 

if __name__ == "__main__":
        # filenames = [os.path.join(SAVE_DIR, filename) for filename in filenames]
    # filenames = find_r0_jsonl_files("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/geeksforgeeks_records")
    filenames = ["/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/geeksforgeeks_records/first_eval/code_translation_GPT-5_geeksforgeeks_optimized_with_extracted_codes_r0-0_geeksforgeeks_formatted_eval.jsonl"]
    # filenames = find_r0_jsonl_files("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/save")
    print(len(filenames))
    print(len(filenames))
    for filename in filenames[:]:
        process_pipeline(filename, k_list=[1], evaluate_datasets=[
                                                                'hackerrank',
                                                                #   'geeksforgeeks'
                                                                  ], 
                                                                  num_workers=4)  # BOTH datasets sequentially, each with 4 workers for smaller chunks
    # get_hackerrank_testcases()

