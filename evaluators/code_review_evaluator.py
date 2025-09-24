"""
Code Review Generation Evaluator

Standalone evaluator that processes raw prediction files from code review generation tasks
and adds evaluation metrics. Follows the pattern: save/ → evaluators → results/

Supports:
- LLM Judge evaluation with configurable models
- Response extraction and parsing for JSON code review responses
- Parallel processing for improved performance
- CodeTREAT format conversion
"""

# Standard library
import os
import json
import re
import copy
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Shared evaluation utilities
from evaluators.utils.utils import (
    get_optimal_worker_count, 
    extract_comment_response, 
    remove_thinking_tag, 
    parse_code_review_outputs
)
from models.model_list import MODELS

# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "code_review_generation")
SAVE_DIR = os.path.join(PROJECT_ROOT, "save")

MODEL_NAME_MAP = {
    'GPT-4o-2024-11-20': 'gpt-4o'
}
def _evaluate_with_llm_judge(generated_review: str, reference_review: str, llm_judge) -> int:
    """Evaluate generated code review using LLM judge."""
    SYSTEM_MESSAGE = (
        "You are a smart code reviewer. You will be asked to grade a generated code review. "
        "You can mimic answering them in the background 10 times and provide me with the most "
        "frequently appearing answer. Furthermore, please strictly adhere to the output format "
        "specified in the question. There is no need to explain your answer. Please output your "
        "final answer in the following JSON format: {\"grade\": <your grade>}. The grade should "
        "be an integer between 1 and 5, inclusive."
    )
    
    JUDGE_PROMPT = (
        "I am going to give you a generated code review as well as its reference review. "
        "You should grade the generated review by comparing it to the reference review, "
        "and output a grade based on the following criteria:\n"
        "1. If the generated review is identical to the reference review, Grade=5;\n"
        "2. If the generated review is essentially equivalent to the reference review although "
        "their expressions are not identical, Grade=4;\n"
        "3. If the generated review explicitly and correctly specifies some comments/suggestions "
        "presented in the reference review, Grade=3;\n"
        "4. If the generated review is only loosely related to the reference review, Grade=2;\n"
        "5. If the generated review is completely unrelated to the reference review in semantics, Grade=1.\n"
        "Please NOTE that you should only output a grade without any explanation.\n"
        "**Generated Code Review**:\n{generated-review}\n"
        "**Reference Code Review**:\n{reference-review}"
    )
    
    judge_prompt = JUDGE_PROMPT.replace("{generated-review}", str(generated_review))
    judge_prompt = judge_prompt.replace("{reference-review}", reference_review)
    
    message = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": judge_prompt}
    ]
    
    # Retry logic for LLM judge
    max_retries = 3
    for attempt in range(max_retries):
        try:
            evaluation = llm_judge.chat(message, n=1, logprobs=False)
            extracted_evaluation = evaluation[0]
            
            # Extract grade from response
            grade_match = re.search(r'"grade":\s*(\d+)', extracted_evaluation)
            if grade_match:
                return int(grade_match.group(1))
                
        except Exception as e:
            print(f"LLM judge attempt {attempt + 1} failed: {e}")
            
    return 1  # Default score if all attempts fail


def _extract_model_name_from_path(filepath: str) -> str:
    """Extract model name from file path."""
    filename = os.path.basename(filepath)
    # Remove .jsonl extension
    model_name = filename.replace('.jsonl', '')
    return model_name


def _detect_file_type(filepath: str, data: List[Dict] = None) -> str:
    """
    Auto-detect if file contains predictions or parsed data.
    
    Returns:
        'predictions' or 'parsed'
    """
    # Check file path first
    if "/predictions/" in filepath:
        return "predictions"
    elif "/parsed/" in filepath:
        return "parsed"
    
    # Check data structure if provided
    if data and len(data) > 0:
        first_item = data[0]
        if 'parsed_response' in first_item:
            return "parsed"
        elif 'response' in first_item and 'parsed_response' not in first_item:
            return "predictions"
    
    # Default assumption - parsed files are more common in evaluation
    return "parsed"


def _convert_predictions_to_parsed(data: List[Dict]) -> List[Dict]:
    """Convert predictions data to parsed format using extraction logic."""
    from extractors.llm_extraction_utils.code_review_extractor import CodeReviewExtractor
    
    extractor = CodeReviewExtractor()
    
    for item in data:
        if 'parsed_response' not in item and 'response' in item:
            responses = item.get('response', [])
            parsed_responses = []
            
            for response in responses:
                extracted = extractor.extract_from_response(response)
                parsed_responses.extend(extracted)
            
            item['parsed_response'] = parsed_responses
    
    return data


def _load_existing_evaluations(evaluation_file: str, judge_model_name: str = None) -> tuple:
    """Load existing evaluations and return set of processed ref_keys for the specific judge model."""
    if not os.path.exists(evaluation_file):
        return set(), []
    
    processed_keys = set()
    existing_data = []
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                existing_data.append(data)
                
                # If judge model specified, only consider items evaluated by that model
                if judge_model_name:
                    if ('ref_key' in data and 'metrics' in data and 
                        MODEL_NAME_MAP[judge_model_name] in data['metrics']):
                        processed_keys.add(data['ref_key'])
                else:
                    # Fallback: consider any item with metrics
                    if 'ref_key' in data and 'metrics' in data:
                        processed_keys.add(data['ref_key'])
            except json.JSONDecodeError:
                continue
    return processed_keys, existing_data


def _filter_unprocessed_items(input_data: List[Dict], processed_keys: set) -> List[Dict]:
    """Filter out items that already have evaluations."""
    return [item for item in input_data if item.get('ref_key') not in processed_keys]


def _process_code_review_item(args):
    """Process a single code review item - thread-safe version."""
    data, max_k, llm_judge = args
    
    try:
        # Get parsed responses (should be done in parsed phase)
        parsed_responses = data.get('parsed_response', [])[:max_k]
        
        if not parsed_responses:
            # Skip items without parsed responses - they should have been processed in parsing phase
            print(f"Warning: No parsed_response found for item {data.get('ref_key', 'unknown')}")
            data['metrics'] = {}
            return data
        
        # Evaluate with LLM judge
        llm_scores = []
        reference_review = data.get('ground_truth', '')
        
        for review in parsed_responses:
            if llm_judge and review and review != "Exceeds Context Length":
                # Extract review content from JSON if needed
                review_text = review
                if isinstance(review, str) and review.startswith('{'):
                    try:
                        import json as json_lib
                        review_json = json_lib.loads(review)
                        review_text = review_json.get('comments', review)
                    except:
                        review_text = review
                
                score = _evaluate_with_llm_judge(str(review_text), reference_review, llm_judge)
                llm_scores.append(score)
            else:
                llm_scores.append(1)  # Default score
        
        # Add evaluation metrics to data with dynamic judge model name
        judge_model_name = getattr(llm_judge, 'model_name', 'gpt-4o-2024-11-20') if llm_judge else 'default'
        
        # Initialize metrics if not exists
        if 'metrics' not in data:
            data['metrics'] = {}
        
        data['metrics'][MODEL_NAME_MAP[judge_model_name]] = llm_scores
        
        # Ensure required fields for results format
        if 'ref_key' not in data:
            # Generate ref_key from available data
            dataset = data.get('dataset', 'github')
            model_name = data.get('model_name', 'unknown')
            prompt_cat = data.get('prompting_category', data.get('prompt_category', 'direct'))
            if isinstance(prompt_cat, list):
                prompt_cat = prompt_cat[0]
            prompt_id = data.get('prompt_id', 1)
            repo_info = f"{data.get('repo', 'unknown')}_{data.get('pr_id', 'unknown')}"
            data['ref_key'] = f"{dataset}_{model_name}_{repo_info}_{prompt_cat}_{prompt_id}"
        
        # Ensure task and dataset fields
        data['task'] = 'code_review_generation'
        if 'dataset' not in data:
            data['dataset'] = 'github'
        
        return data
        
    except Exception as e:
        print(f"Error processing code review item: {str(e)}")
        # Return item with default metrics to avoid losing data
        judge_model_name = getattr(llm_judge, 'model_name', 'gpt-4o-2024-11-20') if llm_judge else 'default'
        
        if 'metrics' not in data:
            data['metrics'] = {}
        data['metrics'][judge_model_name] = [1] * max_k
        
        if 'ref_key' not in data:
            data['ref_key'] = f"unknown_{data.get('model_name', 'unknown')}_unknown_direct_1"
        data['task'] = 'code_review_generation'
        if 'dataset' not in data:
            data['dataset'] = 'github'
        return data


# Note: extract_responses_from_file removed - we work directly with parsed files
# The parsed files already contain extracted_outputs from the parsing phase


def run_llm_evaluation(code_review_data: List[Dict], max_k: int = 1, num_workers: int = None, llm_judge = None) -> List[Dict]:
    """Run LLM evaluation on code review data with parallel processing."""
    print(f"Running LLM evaluation on {len(code_review_data)} items (max_k={max_k})")
    
    # Get LLM judge if not provided
    if llm_judge is None:
        llm_judge = MODELS['gpt-4o-2024-11-20']
        if not llm_judge:
            print("Warning: Could not load LLM judge model, using default scores")
            llm_judge = None
    
    judge_model_name = getattr(llm_judge, 'model_name', 'gpt-4o-2024-11-20') if llm_judge else 'default'
    print(f"Using LLM judge: {judge_model_name}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(4, get_optimal_worker_count(len(code_review_data)))
    else:
        num_workers = min(num_workers, 4)  # Cap at 4 to prevent API rate limiting
    
    print(f"Using {num_workers} workers for evaluation")
    
    # Prepare arguments for each item
    item_args = [(data, max_k, llm_judge) for data in code_review_data]
    
    # Process items in parallel
    evaluated_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_process_code_review_item, item_args),
            total=len(item_args),
            desc="Evaluating code review items"
        ))
        evaluated_data = [result for result in results if result is not None]
    
    print(f"Completed evaluation: {len(evaluated_data)} items processed")
    return evaluated_data


def evaluate_code_review(filename: str, max_k: int = 1, judge_model: str = "gpt-4o-2024-11-20", 
                         auto_parse: bool = None, num_workers: int = None) -> str:
    """Evaluate code review predictions with hybrid approach and breakpoint resuming."""
    print(f"Loading data from: {filename}")
    print(f"Using judge model: {judge_model}")
    
    # Load input data
    code_review_data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            code_review_data.append(data)
    
    print(f"Loaded {len(code_review_data)} items")
    
    # Auto-detect file type if not specified
    if auto_parse is None:
        file_type = _detect_file_type(filename, code_review_data)
        auto_parse = (file_type == "predictions")
        print(f"Auto-detected file type: {file_type}")
    
    # Convert predictions to parsed format if needed
    if auto_parse:
        print("Converting predictions to parsed format...")
        code_review_data = _convert_predictions_to_parsed(code_review_data)
        print("Conversion completed")
    
    # Extract model name and determine paths
    model_name = _extract_model_name_from_path(filename)
    
    # Determine dataset from filename or data
    dataset = "github_2023"  # Default dataset
    if code_review_data and 'dataset' in code_review_data[0]:
        dataset = code_review_data[0]['dataset']
    elif "github" in filename.lower():
        dataset = "github_2023"
    
    # Create evaluation directory and file path
    evaluation_dir = os.path.join(RESULTS_DIR, dataset, "evaluations")
    os.makedirs(evaluation_dir, exist_ok=True)
    evaluation_file = os.path.join(evaluation_dir, f"{model_name}.jsonl")
    
    # Get judge model from MODELS dict
    llm_judge = MODELS.get(judge_model)
    if not llm_judge:
        print(f"Warning: Judge model {judge_model} not found in MODELS, using default scores")
        llm_judge = None
    
    judge_model_name = getattr(llm_judge, 'model_name', judge_model) if llm_judge else judge_model
    
    # Load existing evaluations for breakpoint resuming
    processed_keys, existing_data = _load_existing_evaluations(evaluation_file, judge_model_name)
    print(f"Found {len(processed_keys)} items already evaluated with {judge_model_name}")
    
    # Filter unprocessed items
    unprocessed_data = _filter_unprocessed_items(code_review_data, processed_keys)
    print(f"Processing {len(unprocessed_data)} new items")
    
    if not unprocessed_data:
        print("All items already evaluated!")
        return evaluation_file
    
    # Run evaluation on unprocessed items
    evaluated_data = run_llm_evaluation(unprocessed_data, max_k, num_workers, llm_judge)
    
    # Append new evaluations to existing file
    with open(evaluation_file, "a", encoding="utf-8") as f:
        for d in evaluated_data:
            f.write(json.dumps(d) + '\n')
    
    total_items = len(existing_data) + len(evaluated_data)
    print(f"Code review evaluation completed: {len(evaluated_data)} new items evaluated")
    print(f"Total items in evaluation file: {total_items}")
    print(f"Results saved to: {evaluation_file}")
    return evaluation_file


def process_pipeline(filename: str, max_k: int = 1, judge_model: str = "gpt-4o-2024-11-20", 
                    auto_parse: bool = None, extract_only: bool = False, num_workers: int = None) -> str:
    """Main processing pipeline for code review evaluation with hybrid approach."""
    print(f"Processing {filename}")
    
    try:
        if extract_only:
            # Create separate extractor instance for extraction only
            from extractors.llm_extraction_utils.code_review_extractor import CodeReviewExtractor
            extractor = CodeReviewExtractor()
            return extractor.extract(filename)
        
        # Hybrid evaluation - supports both predictions and parsed files
        evaluation_file = evaluate_code_review(filename, max_k, judge_model, auto_parse, num_workers)
        
        print(f"Pipeline completed: {evaluation_file}")
        return evaluation_file
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise


def evaluate_code_review_file(filename: str, max_k: int = 1, judge_model: str = "gpt-4o-2024-11-20", 
                             auto_parse: bool = None, extract_only: bool = False, num_workers: int = None) -> str:
    """
    Convenience function to evaluate a single code review file.
    
    Args:
        filename: Path to predictions or parsed JSONL file
        max_k: Number of responses to evaluate per item
        judge_model: LLM judge model name (from MODELS dict)
        auto_parse: Force parsing mode (None=auto-detect, True=parse predictions, False=use parsed)
        extract_only: Only extract to parsed format (no evaluation)
        num_workers: Number of parallel workers for evaluation
    
    Returns:
        Path to evaluation output file (or parsed file if extract_only=True)
    
    Examples:
        # Work from parsed files (extractor approach)
        evaluate_code_review_file("results/.../parsed/GPT-5.jsonl", judge_model="gpt-4o-2024-11-20")
        
        # Work from predictions with internal parsing (one-click)  
        evaluate_code_review_file("results/.../predictions/GPT-5.jsonl", judge_model="gpt-4o-2024-11-20", auto_parse=True)
    """
    return process_pipeline(filename, max_k=max_k, judge_model=judge_model, auto_parse=auto_parse,
                          extract_only=extract_only, num_workers=num_workers)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python code_review_evaluator.py <file> [max_k] [--judge-model MODEL] [--auto-parse] [--extract-only]")
        print("Examples:")
        print("  # Work from parsed files (extractor approach)")
        print("  python code_review_evaluator.py results/.../parsed/GPT-5.jsonl 1 --judge-model gpt-4o-2024-11-20")
        print("  # Work from predictions with internal parsing (one-click)")
        print("  python code_review_evaluator.py results/.../predictions/GPT-5.jsonl 1 --judge-model gpt-4o-2024-11-20 --auto-parse")
        sys.exit(1)
    
    filename = sys.argv[1]
    extract_only = "--extract-only" in sys.argv
    auto_parse_flag = "--auto-parse" in sys.argv
    
    # Parse max_k value
    max_k = 1  # Default
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg.isdigit():
            max_k = int(arg)
            break
    
    # Parse judge model
    judge_model = "gpt-4o-2024-11-20"  # Default
    if "--judge-model" in sys.argv:
        judge_idx = sys.argv.index("--judge-model")
        if judge_idx + 1 < len(sys.argv):
            judge_model = sys.argv[judge_idx + 1]
    
    # Set auto_parse parameter
    auto_parse = True if auto_parse_flag else None  # None = auto-detect
    
    print(f"Processing {filename} with max_k={max_k}, judge_model={judge_model}")
    if extract_only:
        print("Extract only mode - no LLM evaluation")
    elif auto_parse_flag:
        print("Auto-parse mode - will parse predictions internally")
    else:
        print("Auto-detect mode - will detect file type automatically")
    
    result_file = process_pipeline(filename, max_k=max_k, judge_model=judge_model, 
                                 auto_parse=auto_parse, extract_only=extract_only)
    print(f"Processing complete: {result_file}")