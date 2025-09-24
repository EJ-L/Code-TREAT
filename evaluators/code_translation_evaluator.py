"""
Code Translation Evaluator

Standalone evaluator that processes raw prediction files from code translation tasks
and adds evaluation metrics. Follows the pattern: save/ → evaluators → results/

Supports:
- HackerRank benchmark evaluation with test case execution
- PolyHumanEval benchmark evaluation with project template system
- Pass@k evaluation metrics
- Parallel processing for improved performance
"""

# Standard library
import os
import json
import tempfile
from typing import List, Dict, Any
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# Shared evaluation utilities
from evaluators.utils import (
    CodeExecutor,
    decompress_test_cases,
    compute_pass_at_k_from_results,
    get_optimal_worker_count,
)
from evaluators.utils.python_version_manager import PythonVersionManager

# Tree-sitter utilities for code formatting
from extractors.tree_sitter_extraction_utils.python_tree_sitter_utils import python_polyhumaneval_formatter, clean_if_name
from extractors.tree_sitter_extraction_utils.java_tree_sitter_utils import java_polyhumaneval_formatter, remove_java_imports, JAVA_BASE_IMPORTS

# Benchmark evaluation pipeline
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.parsing import parse
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.eval import ProjectTemplate, EvalStatus, gen_codes_for_single_file, create_project
POLYHUMANEVAL_AVAILABLE = True

def get_geeksforgeeks_testcases() -> Dict[int, List[Dict[str, str]]]:
    testcase_hash = {}
    driver_hash = {}
    func_sign_info = {}
    user_hash = {}
    with open("/Users/ericjohnli/Downloads/TREAT-refined/data/geeksforgeeks/geeksforgeeks_filtered.jsonl", "r", encoding="utf-8") as f:
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

### imports that may be needed for python

# Global executor instance - use Python 3.9 for better compatibility
# Set up Python version with automatic detection and validation
python_manager = PythonVersionManager()
PYTHON_EXECUTABLE, setup_messages = python_manager.setup_environment()
# print(setup_messages)
EXECUTOR = CodeExecutor(python_executable=PYTHON_EXECUTABLE)
INFO_HASH = get_geeksforgeeks_testcases()
# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "code_translation")

# Dataset-specific paths
HACKERRANK_TESTCASE_PATH = os.path.join(
    PROJECT_ROOT, "data", "hackerrank", "hackerrank_filtered.jsonl"
)
POLYHUMANEVAL_TESTCASE_PATH = os.path.join(
    PROJECT_ROOT, "benchmark_modules", "polyhumaneval_benchmark", "evaluation", "data", "poly_humanevalv4.testdsl"
)
POLYHUMANEVAL_FUNC_REF_PATH = os.path.join(
    PROJECT_ROOT, "benchmark_modules", "polyhumaneval_benchmark", "evaluation", "data", "polyhumaneval_func_names.json"
)

def get_hackerrank_testcases() -> Dict[int, List[Dict[str, str]]]:
    testcase_hash = {}
    with open(HACKERRANK_TESTCASE_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            testcase_hash[data['question_id']] = decompress_test_cases(data['test_cases'])
    return testcase_hash

def get_polyhumaneval_test_inputs() -> Dict[str, str]:
    if not POLYHUMANEVAL_AVAILABLE:
        return {}, {}
        
    try:
        question_id_to_test_input = {}
        with open(POLYHUMANEVAL_TESTCASE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            last_problem_idx = 0
            problem_id = None
            for idx, line in enumerate(lines):
                if "problem " in line:
                    if problem_id is not None:  # Ensure this is not the first problem found
                        question_id_to_test_input[problem_id] = ''.join(lines[last_problem_idx:idx])
                    problem_id = line.split("problem ")[1].split(" ")[0]
                    last_problem_idx = idx
                    # Handle the last problem if any
                    if problem_id is not None:
                        question_id_to_test_input[problem_id] = ''.join(lines[last_problem_idx:])

        with open(POLYHUMANEVAL_FUNC_REF_PATH, "r", encoding="utf-8") as f:
            print(POLYHUMANEVAL_FUNC_REF_PATH)
            func_ref = json.load(f)
            
        return question_id_to_test_input, func_ref
    except Exception as e:
        print(f"Warning: Could not load PolyHumanEval test inputs: {e}")
        return {}, {}

POLYHUMANEVAL_TEST_INPUT_HASH, POLYHUMANEVAL_FUNC_REF = get_polyhumaneval_test_inputs()

def _process_hackerrank_item(args):
    """Process a single HackerRank item - thread-safe version."""
    data, k_list, testcase_hash = args
    
    try:
        _id = data['id']
        language = data['target_lang']
        predictions = data['parsed_response'][:min(max(k_list), len(data['parsed_response']))]
        
        if _id not in testcase_hash:
            data['executed_code'] = []
            data['results'] = [{'error': 'not test cases'}]
            data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
            return data
            
        # Create fresh executors for this thread to avoid contention
        test_cases = testcase_hash[_id]
        test_inputs = [test_case['input'] for test_case in test_cases]
        test_outputs = [test_case['expected_output'] for test_case in test_cases]
        
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
        print(f"Error processing Hackerrank item {language} {_id}: {str(e)}")
        data['executed_code'] = []
        data['evaluation_results'] = [{'error': str(e)}]
        data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
        return data

def run_hackerrank(hackerrank_data: Dict[str, Any], k_list: List[int], num_workers: int = None) -> Dict[str, Any]:
    """Improved parallel HackerRank evaluation with item-level parallelization."""
    print(f"Starting parallel HackerRank evaluation with {len(hackerrank_data)} items")
    
    # Get test cases once (shared across all workers)
    testcase_hash = get_hackerrank_testcases()
    
    # Determine optimal worker count - limit to avoid overwhelming system
    if num_workers is None:
        num_workers = min(8, get_optimal_worker_count(len(hackerrank_data)))
    else:
        num_workers = min(num_workers, 8)  # Cap at 4 to prevent resource issues
    
    print(f"Using {num_workers} workers for HackerRank evaluation")
    
    # Prepare arguments for each item
    item_args = [(data, k_list, testcase_hash) for data in hackerrank_data]
    
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

def _process_polyhumaneval_item(args):
    """Process a single PolyHumanEval item in a worker process."""
    data, k_list, test_input_hash = args
    
    try:
        _id = data['id']
        language = data['target_lang']
        predictions = data['parsed_response'][:min(max(k_list), len(data['parsed_response']))]
        # if 'metrics' in data and data['metrics']['pass@1'] == 1.0:
        #     return data
        # Parse test input in worker process to avoid serialization issues
        problems = parse(test_input_hash[_id])
        results = []
        executed_codes = []
        # Process predictions for this item
        for _, code in enumerate(predictions):
            # Use tempfile for better cleanup
            with tempfile.TemporaryDirectory() as temp_dir:
                imports = ""
                try:
                    template = ProjectTemplate(f"{PROJECT_ROOT}/benchmark_modules/polyhumaneval_benchmark/evaluation/project-templates/default/{language}")
                    if language == 'java':
                        code, import_list = remove_java_imports(code)
                        code = java_polyhumaneval_formatter(code, POLYHUMANEVAL_FUNC_REF[language][_id])
                        imports = '\n'.join(import_list) if import_list else ""
                        imports += JAVA_BASE_IMPORTS
                    
                    if language == 'python':
                        code = python_polyhumaneval_formatter(code, POLYHUMANEVAL_FUNC_REF[language][_id])
                        code = clean_if_name(code)
                        
                    executed_codes.append(code)
                    codes = gen_codes_for_single_file(
                        lang=language, 
                        problem=list(problems.values())[0], 
                        target_code=code
                    )
                    proj = create_project(
                        template=template, 
                        name=f"{temp_dir}", 
                        codes=codes, 
                        root=temp_dir, 
                        overwrite=True,
                        imports=imports
                    )

                    ret_stat, msg = proj.evaluate(
                        compile_timeout=300,
                        run_timeout=300,
                        keep_after_eval=False
                    )
                    
                    if ret_stat == EvalStatus.Pass:
                        results.append({"success": True, "msg": msg})
                    else:
                        results.append({"success": False, "msg": msg})
                except Exception as e:
                    results.append({"success": False, "msg": str(e)})
        data['executed_code'] = executed_codes
        data['evaluation_results'] = results
        data['metrics'] = compute_pass_at_k_from_results(results, k_list)
        return data
        
    except Exception as e:
        print(f"Error processing PolyHumanEval item {language} {_id}: {str(e)}")
        data['executed_code'] = []
        data['evaluation_results'] = [{'error': str(e)}]
        data['metrics'] = {f'pass@{k}': 0.0 for k in k_list if k >= len(data['parsed_response'])}
        return data

def run_polyhumaneval(polyhumaneval_data: Dict[str, Any], k_list: int, num_workers: int = None) -> Dict[str, Any]:
    """
    Parallel PolyHumanEval evaluation using process-based multiprocessing.
    
    Key insight: Parse test inputs in each worker process to avoid lambda function 
    serialization issues with threading/multiprocessing.
    """
    print(f"Starting parallel PolyHumanEval evaluation with {len(polyhumaneval_data)} items")
    
    # Determine optimal worker count
    if num_workers is None:
        num_workers = get_optimal_worker_count(len(polyhumaneval_data))
    
    print(f"Using {num_workers} workers for PolyHumanEval evaluation")
    
    # Prepare arguments for each item
    item_args = [
        (data, k_list, POLYHUMANEVAL_TEST_INPUT_HASH) 
        for data in polyhumaneval_data
    ]
    
    # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid serialization issues
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process with progress bar
        results = list(tqdm(
            executor.map(_process_polyhumaneval_item, item_args),
            total=len(item_args),
            desc="Processing PolyHumanEval items",
            unit="item"
        ))
    
    print(f"Completed PolyHumanEval evaluation: {len(results)} items processed")
    return results

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

def evaluate_polyhumaneval(filename: str, k_list: List[int], num_workers: int = None) -> str:
    polyhumaneval_data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            data = json.loads(line)
            polyhumaneval_data.append(data)
    
    evaluated_data = run_polyhumaneval(polyhumaneval_data, k_list, num_workers)
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
        if 'polyhumaneval' in evaluate_datasets:
            # Run PolyHumanEval second
            evaluate_polyhumaneval(filename, k_list, num_workers)
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        # Don't remove files if there was an error
        raise 

if __name__ == "__main__":
    
    
    POLYHUMANEVAL_TEST_INPUT_HASH, POLYHUMANEVAL_FUNC_REF = get_polyhumaneval_test_inputs()
    # filenames = [os.path.join(SAVE_DIR, filename) for filename in filenames]
    # filenames = find_r0_jsonl_files("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/polyhumaneval_records")
    filenames = ["/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/polyhumaneval_records/first_eval/code_translation_GPT-5_polyhumaneval_optimized_with_extracted_codes_r0-0_polyhumaneval_formatted_eval.jsonl"]
    # filenames = find_r0_jsonl_files("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_translation/save")
    print(len(filenames))
    print(len(filenames))
    for filename in filenames[:]:
        process_pipeline(filename, k_list=[1], evaluate_datasets=[
                                                                'hackerrank', 
                                                                #   'polyhumaneval'
                                                                  ], 
                                                                  num_workers=4)  # BOTH datasets sequentially, each with 4 workers for smaller chunks
    # get_hackerrank_testcases()

