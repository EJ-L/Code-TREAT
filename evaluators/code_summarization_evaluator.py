"""
Code Summarization Evaluator

Standalone evaluator that processes prediction files from code summarization tasks
and adds evaluation metrics. Supports hybrid workflow with breakpoint resuming.

Supports:
- Hybrid approach: auto-detection of predictions vs parsed files
- Breakpoint resuming to avoid expensive LLM re-evaluation
- Configurable LLM Judge models for cost optimization
- Auto-parsing with internal extraction or separate extractor workflow
- Parallel processing for improved performance
"""

# Standard library
import os
import json
import re
import copy
import glob
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Shared evaluation utilities
from evaluators.utils.utils import get_optimal_worker_count, remove_thinking_tag, parse_response_outputs, extract_evaluation_json
from extractors.llm_extraction_utils.code_summarization_extractor import CodeSummarizationExtractor
from models.model_list import MODELS

# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "code_summarization")
SAVE_DIR = os.path.join(PROJECT_ROOT, "save")

def _evaluate_with_llm_judge(generated_comment: str, reference_comment: str, llm_judge, judge_model_name: str = "gpt-4o-2024-11-20") -> int:
    """Evaluate generated comment using LLM judge."""
    SYSTEM_MESSAGE = (
        "You are a smart code reviewer. You will be asked to grade a generated code comment. "
        "You can mimic answering them in the background 10 times and provide me with the most "
        "frequently appearing answer. Furthermore, please strictly adhere to the output format "
        "specified in the question. There is no need to explain your answer. Please output your "
        "final answer in the following JSON format: {\"grade\": <your grade>}. The grade should "
        "be an integer between 1 and 5, inclusive."
    )
    
    JUDGE_PROMPT = (
        "I am going to give you a generated code comment as well as its reference comment. "
        "You should grade the generated comment by comparing it to the reference comment, "
        "and output a grade based on the following criteria:\n"
        "1. If the generated comment is identical to the reference comment, Grade=5;\n"
        "2. If the generated comment is essentially equivalent to the reference comment although "
        "their expressions are not identical, Grade=4;\n"
        "3. If the generated comment explicitly and correctly specifies some aspects "
        "presented in the reference comment, Grade=3;\n"
        "4. If the generated comment is only loosely related to the reference comment, Grade=2;\n"
        "5. If the generated comment is completely unrelated to the reference comment in semantics, Grade=1.\n"
        "Please NOTE that you should only output a grade without any explanation.\n"
        "**Generated Code Comment**:\n{generated-comment}\n"
        "**Reference Code Comment**:\n{reference-comment}"
    )
    
    judge_prompt = JUDGE_PROMPT.replace("{generated-comment}", str(generated_comment))
    judge_prompt = judge_prompt.replace("{reference-comment}", reference_comment)
    
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
            
    print(f"Warning: All LLM judge attempts failed for model {judge_model_name}, using default score")
    return 1  # Default score if all attempts fail


def _detect_file_type(filepath: str, data: List[Dict] = None) -> str:
    """Auto-detect if file contains predictions or parsed data."""
    # Check file path first
    if "/predictions/" in filepath:
        return "predictions"
    elif "/parsed/" in filepath:
        return "parsed"
    
    # Check data content if provided
    if data:
        sample = data[0] if data else {}
        if "parsed_response" in sample:
            return "parsed"
        elif "response" in sample or "outputs" in sample:
            return "predictions"
    
    # Default assumption
    return "predictions"


def _load_existing_evaluations(evaluation_file: str, judge_model_name: str) -> Dict[str, Dict]:
    """Load existing evaluation results for breakpoint resuming."""
    if not os.path.exists(evaluation_file):
        return {}
    
    existing_evaluations = {}
    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ref_key = data.get('ref_key')
                    if ref_key and 'metrics' in data:
                        metrics = data.get('metrics', {})
                        if judge_model_name in metrics:
                            existing_evaluations[ref_key] = data
    except Exception as e:
        print(f"Warning: Could not load existing evaluations: {e}")
    
    return existing_evaluations


def extract_base_key_from_ref_key(ref_key: str) -> str:
    """
    Extract base grouping key from ref_key for batch evaluation.
    
    Converts:
    - github_2023_gpt_5_repo_info_direct_1 -> github_2023_repo_info_direct
    - github_2023_baseline_repo_info_direct_X -> github_2023_repo_info_direct
    """
    parts = ref_key.split('_')
    if len(parts) < 4:
        return ref_key  # Fallback for malformed keys
    
    # Strategy: Remove model name and prompt_id to get base key
    # Model name typically starts after dataset (parts[2] onwards)
    # Prompt_id is always the last part
    
    # Find where the model name ends by looking for common patterns
    # Baseline case: github_2023_baseline_repo_info_direct_X
    # Model case: github_2023_gpt_5_repo_info_direct_1
    
    if parts[2] == 'baseline':
        # Baseline: remove 'baseline' and last part (prompt_id)
        base_parts = [parts[0], parts[1]] + parts[3:-1]
    else:
        # Regular model: need to find where model name ends
        # Look for known model patterns or use heuristic
        # For now, assume model name is parts[2] (possibly with parts[3] if it's a number)
        if len(parts) > 3 and parts[3].isdigit():
            # Model name includes number: gpt_5, qwen_3, etc.
            base_parts = [parts[0], parts[1]] + parts[4:-1]
        else:
            # Simple model name: gpt, claude, etc.
            base_parts = [parts[0], parts[1]] + parts[3:-1]
    
    return '_'.join(base_parts)


def collect_for_batch_evaluation(dataset_dir: str) -> Dict[str, Dict]:
    """
    Collect all model responses grouped by base_key for batch evaluation.
    
    Returns:
        Dict[base_key, {
            'baseline': baseline_item or None,
            'models': {model_name: {prompt_id: item}},
            'metadata': {code/function, url, lang}
        }]
    """
    print(f"Collecting data for batch evaluation from: {dataset_dir}")
    
    groups = defaultdict(lambda: {
        'baseline': None,
        'models': defaultdict(dict),
        'metadata': {}
    })
    
    # Find all parsed files
    parsed_dir = os.path.join(dataset_dir, "parsed")
    if not os.path.exists(parsed_dir):
        print(f"Warning: Parsed directory not found: {parsed_dir}")
        return {}
    
    parsed_files = glob.glob(os.path.join(parsed_dir, "*.jsonl"))
    print(f"Found {len(parsed_files)} parsed files")
    
    for parsed_file in parsed_files:
        model_name = os.path.basename(parsed_file).replace('.jsonl', '')
        print(f"Processing {model_name}...")
        
        with open(parsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                base_key = extract_base_key_from_ref_key(item['ref_key'])
                
                if model_name == 'Baseline':
                    groups[base_key]['baseline'] = item
                else:
                    prompt_id = item['prompt_id']
                    groups[base_key]['models'][model_name][prompt_id] = item
                
                # Store metadata once per group
                if not groups[base_key]['metadata']:
                    groups[base_key]['metadata'] = {
                        'code/function': item['code/function'],
                        'url': item['url'],
                        'lang': item['lang'],
                        'dataset': item.get('dataset', 'unknown')
                    }
    
    print(f"Collected {len(groups)} unique code/prompt groups")
    return dict(groups)


def load_evaluation_cache(cache_file: str) -> Dict:
    """Load existing evaluation results to prevent re-evaluation."""
    if not cache_file or not os.path.exists(cache_file):
        return {}
    
    cache = {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_record = json.loads(line)
                    # Use base_key as cache key
                    base_key = eval_record.get('base_key')
                    if base_key:
                        cache[base_key] = eval_record
    except Exception as e:
        print(f"Warning: Could not load evaluation cache: {e}")
    
    return cache


def save_evaluation_cache(cache_file: str, evaluation_results: Dict) -> None:
    """Save evaluation results to cache file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        for base_key, eval_data in evaluation_results.items():
            eval_record = eval_data.copy()
            eval_record['base_key'] = base_key
            f.write(json.dumps(eval_record) + '\n')


def build_judge_prompt(group_data: Dict, prompt_id: int, comment_model_map: Dict) -> str:
    """Build judge prompt following old TREAT pattern."""
    
    JUDGE_PROMPT = """Here is a piece of code with corresponding comments. Please rate each comment on a scale from 1 to 5, where a higher score indicates better quality. A good comment should: 1) accurately summarize the function of the code; 2) be expressed naturally and concisely, without burdening the developer with reading; 3) help the developer understand the code quickly: Your answer should be in the JSON format JSON: {'Comment 0': {your rating}, 'Comment 1': {your rating}, ..., 'Comment n': {your rating}}.
Code:
"""
    
    current_judge_prompt = JUDGE_PROMPT
    current_judge_prompt += f"{group_data['metadata']['code/function']}\n"
    
    # Add baseline comment (Comment 0) if available
    if group_data['baseline'] and 'Comment 0' in comment_model_map:
        if 'parsed_response' in group_data['baseline']:
            baseline_comment = group_data['baseline']['parsed_response'][0]
        else:
            # Fallback to response field
            baseline_comment = group_data['baseline']['response']
            if isinstance(baseline_comment, list):
                baseline_comment = baseline_comment[0]
        current_judge_prompt += f"Comment 0: {baseline_comment}\n"
    
    # Add model comments (Comment 1, 2, 3...)
    comment_idx = 1
    for model_name in group_data['models']:
        if prompt_id in group_data['models'][model_name]:
            item = group_data['models'][model_name][prompt_id]
            if 'parsed_response' in item and item['parsed_response']:
                comment = item['parsed_response'][0]  # Take first response
            else:
                # Fallback to response field
                comment = item.get('response', [''])[0] if isinstance(item.get('response'), list) else item.get('response', '')
            
            current_judge_prompt += f"Comment {comment_idx}: {comment}\n"
            comment_idx += 1
    
    return current_judge_prompt


def call_llm_judge(llm_judge, judge_prompt: str) -> Dict:
    """Call LLM judge and extract JSON evaluation."""
    judge_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": judge_prompt}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            evaluation_result = llm_judge.chat(judge_message)[0]
            json_result = extract_evaluation_json(evaluation_result)
            
            if json_result is not None:
                return json.loads(json_result)
            else:
                print(f"Could not extract JSON from evaluation result, attempt {attempt + 1}")
                continue
                
        except Exception as e:
            print(f"LLM judge call failed, attempt {attempt + 1}: {e}")
    
    # Return default scores if all attempts fail
    print("Warning: All LLM judge attempts failed, using default scores")
    return {"Comment 0": 1, "Comment 1": 1, "Comment 2": 1, "Comment 3": 1}


def batch_llm_evaluate(groups: Dict, judge_model: str = 'gpt-4o-2024-11-20', 
                      eval_cache_file: str = None) -> Dict:
    """
    Batch evaluate using old TREAT pattern: single LLM call evaluates all models' 
    responses for same code/prompt combination.
    """
    print(f"Starting batch evaluation with judge model: {judge_model}")
    
    # Load existing evaluation cache
    eval_cache = load_evaluation_cache(eval_cache_file) if eval_cache_file else {}
    print(f"Loaded {len(eval_cache)} cached evaluations")
    
    llm_judge = MODELS.get(judge_model)
    if not llm_judge:
        print(f"Warning: Could not load judge model {judge_model}")
        return {}
    
    evaluation_results = {}
    new_evaluations = 0
    
    for base_key, group_data in tqdm(groups.items(), desc="Batch evaluating groups"):
        if base_key in eval_cache:
            evaluation_results[base_key] = eval_cache[base_key]
            continue
        
        # Build comment model mapping
        comment_model_map = {}
        if group_data['baseline']:
            comment_model_map['Comment 0'] = 'Baseline'
        
        comment_idx = 1
        for model_name in sorted(group_data['models'].keys()):
            comment_model_map[f'Comment {comment_idx}'] = model_name
            comment_idx += 1
        
        # Evaluate each prompt_id (1, 2, 3)
        prompt_evaluations = {}
        for prompt_id in [1, 2, 3]:
            # Check if we have data for this prompt_id
            has_data = any(prompt_id in group_data['models'][model] for model in group_data['models'])
            if not has_data:
                continue
                
            judge_prompt = build_judge_prompt(group_data, prompt_id, comment_model_map)
            evaluation_result = call_llm_judge(llm_judge, judge_prompt)
            prompt_evaluations[prompt_id] = evaluation_result
        
        evaluation_results[base_key] = {
            'comment_model_map': comment_model_map,
            'prompt_evaluations': prompt_evaluations,
            'judge_model': judge_model,
            'metadata': group_data['metadata']
        }
        new_evaluations += 1
    
    print(f"Completed batch evaluation: {new_evaluations} new evaluations")
    
    # Save to cache
    if eval_cache_file and new_evaluations > 0:
        save_evaluation_cache(eval_cache_file, evaluation_results)
        print(f"Saved evaluation cache to: {eval_cache_file}")
    
    return evaluation_results


def distribute_evaluation_results(groups: Dict, evaluation_results: Dict, dataset_dir: str) -> None:
    """
    Distribute batch evaluation results back to individual model files.
    
    Creates evaluation files in dataset_dir/evaluations/ with scores added to each item.
    """
    print("Distributing evaluation results to individual model files...")
    
    evaluation_dir = os.path.join(dataset_dir, "evaluations")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Track which models we need to process
    all_models = set()
    for group_data in groups.values():
        all_models.update(group_data['models'].keys())
        if group_data['baseline']:
            all_models.add('Baseline')
    
    print(f"Processing {len(all_models)} models")
    
    for model_name in all_models:
        print(f"Processing {model_name}...")
        
        # Load all items for this model from parsed file
        parsed_file = os.path.join(dataset_dir, "parsed", f"{model_name}.jsonl")
        if not os.path.exists(parsed_file):
            print(f"Warning: Parsed file not found for {model_name}")
            continue
        
        updated_items = []
        
        with open(parsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                base_key = extract_base_key_from_ref_key(item['ref_key'])
                
                # Find evaluation data for this item
                if base_key in evaluation_results:
                    eval_data = evaluation_results[base_key]
                    comment_model_map = eval_data['comment_model_map']
                    judge_model = eval_data['judge_model']
                    
                    # Find which comment index this model corresponds to
                    model_comment_id = None
                    for comment_id, mapped_model in comment_model_map.items():
                        if mapped_model == model_name:
                            model_comment_id = comment_id
                            break
                    
                    if model_comment_id:
                        # Get evaluation score for this prompt_id
                        if model_name == 'Baseline':
                            # Baseline: map to each prompt_id (1, 2, 3)
                            prompt_id = item.get('prompt_id', 1)
                            if prompt_id == 'X':
                                # Create separate entries for each prompt_id
                                for pid in [1, 2, 3]:
                                    baseline_item = item.copy()
                                    baseline_item['prompt_id'] = pid
                                    baseline_item['ref_key'] = baseline_item['ref_key'].replace('_X', f'_{pid}')
                                    
                                    if pid in eval_data['prompt_evaluations']:
                                        score = eval_data['prompt_evaluations'][pid].get(model_comment_id, 1)
                                        if 'metrics' not in baseline_item:
                                            baseline_item['metrics'] = {}
                                        baseline_item['metrics'][judge_model] = score
                                    
                                    updated_items.append(baseline_item)
                                # IMPORTANT: Skip adding original item with prompt_id='X'
                                continue
                            else:
                                # Normal baseline item with specific prompt_id
                                prompt_id = item['prompt_id']
                        else:
                            # Regular model
                            prompt_id = item['prompt_id']
                        
                        # Add evaluation score (only for non-X baseline items and regular models)
                        if prompt_id in eval_data['prompt_evaluations']:
                            score = eval_data['prompt_evaluations'][prompt_id].get(model_comment_id, 1)
                            if 'metrics' not in item:
                                item['metrics'] = {}
                            item['metrics'][judge_model] = score
                        else:
                            print(f"Warning: No evaluation found for {model_name} prompt_id {prompt_id}")
                
                # Add item to results (this will be skipped for baseline items with prompt_id='X' due to continue above)
                updated_items.append(item)
        
        # Write updated file to evaluations directory
        eval_file = os.path.join(evaluation_dir, f"{model_name}.jsonl")
        with open(eval_file, 'w', encoding='utf-8') as f:
            for item in updated_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  Wrote {len(updated_items)} items to {eval_file}")
    
    print("✅ Distribution complete!")


def batch_evaluate_dataset(dataset_dir: str, judge_model: str = 'gpt-4o-2024-11-20') -> str:
    """
    Main function to perform batch evaluation following old TREAT pattern.
    
    Args:
        dataset_dir: Path to dataset directory (e.g., results/code_summarization/github_2023)
        judge_model: LLM judge model name
        
    Returns:
        Path to the evaluation cache file
    """
    print(f"=== Batch Code Summarization Evaluation ===")
    print(f"Dataset: {dataset_dir}")
    print(f"Judge Model: {judge_model}")
    
    # Step 1: Collect all model responses by base_key
    groups = collect_for_batch_evaluation(dataset_dir)
    if not groups:
        print("No data collected for evaluation")
        return None
    
    # Step 2: Batch LLM evaluation with caching
    cache_file = os.path.join(dataset_dir, f"evaluation_cache_{judge_model.replace('/', '_')}.jsonl")
    evaluation_results = batch_llm_evaluate(groups, judge_model, cache_file)
    
    # Step 3: Distribute results back to individual model files
    distribute_evaluation_results(groups, evaluation_results, dataset_dir)
    
    print(f"✅ Batch evaluation completed!")
    return cache_file


def _convert_predictions_to_parsed(data: List[Dict]) -> List[Dict]:
    """Convert predictions format to parsed format with parsed_response field."""
    print("Converting predictions to parsed format...")
    converted_data = []
    
    for item in data:
        # Create a copy to avoid modifying original data
        converted_item = item.copy()
        
        # Extract responses using the same logic as extractor
        responses = item.get('response', item.get('outputs', []))
        lang = item.get('lang', 'python')
        
        parsed_responses = []
        for response in responses:
            if not response or response == "Exceeds Context Length":
                parsed_responses.append(response if response else "")
                continue
            
            # Remove thinking tags and parse
            cleaned_response = remove_thinking_tag(response)
            parsed_outputs = parse_response_outputs([cleaned_response], lang)
            parsed_responses.extend(parsed_outputs)
        
        # Add parsed_response field
        converted_item['parsed_response'] = parsed_responses
        converted_data.append(converted_item)
    
    print("Conversion completed")
    return converted_data


def _process_code_summarization_item(args):
    """Process a single code summarization item - thread-safe version."""
    data, llm_judge, judge_model_name = args
    
    try:
        # Get parsed responses (framework standard)
        parsed_responses = data.get('parsed_response', [])
        if not parsed_responses:
            # Fallback for legacy format
            parsed_responses = data.get('extracted_outputs', [])
        
        # Evaluate with LLM judge
        llm_scores = []
        reference_comment = data.get('ground_truth', data.get('parsed_docstring', ''))
        
        for comment in parsed_responses:
            if llm_judge and comment and comment != "Exceeds Context Length":
                score = _evaluate_with_llm_judge(comment, reference_comment, llm_judge, judge_model_name)
                llm_scores.append(score)
            else:
                llm_scores.append(1)  # Default score
        
        # Add evaluation metrics to data using dynamic judge model name
        if 'metrics' not in data:
            data['metrics'] = {}
        data['metrics'][judge_model_name] = llm_scores
        
        # Ensure required fields for results format
        if 'ref_key' not in data:
            # Generate ref_key from available data
            dataset = data.get('dataset', 'general')
            model_name = data.get('model_name', 'unknown')
            prompt_cat = data.get('prompting_category', data.get('prompt_category', ['direct']))[0] if isinstance(data.get('prompting_category', data.get('prompt_category', ['direct'])), list) else data.get('prompting_category', data.get('prompt_category', 'direct'))
            prompt_id = data.get('prompt_id', 1)
            code_snippet = data.get('code/function', data.get('url', 'unknown'))[:50].replace('/', '_').replace(' ', '_')
            data['ref_key'] = f"{dataset}_{model_name}_{code_snippet}_{prompt_cat}_{prompt_id}"
        
        # Ensure task and dataset fields
        data['task'] = 'code_summarization'
        if 'dataset' not in data:
            data['dataset'] = 'general'
        
        return data
        
    except Exception as e:
        print(f"Error processing code summarization item: {str(e)}")
        # Return item with default metrics to avoid losing data
        if 'metrics' not in data:
            data['metrics'] = {}
        data['metrics'][judge_model_name] = [1]
        if 'ref_key' not in data:
            data['ref_key'] = f"unknown_{data.get('model_name', 'unknown')}_unknown_direct_1"
        data['task'] = 'code_summarization'
        if 'dataset' not in data:
            data['dataset'] = 'general'
        return data


def extract_responses_from_file(filename: str) -> str:
    """Extract responses from a single prediction file."""
    print(f"Extracting responses from {filename}")
    new_json_lines = []
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            
            # Remove thinking tags and parse responses
            outputs = [remove_thinking_tag(output) for output in json_line["outputs"]]
            extracted_outputs = parse_response_outputs(outputs, json_line["lang"])
            
            json_line["extracted_outputs"] = extracted_outputs
            new_json_lines.append(json_line)
    
    # Write extracted responses
    extracted_filename = filename.replace('.jsonl', '_extracted.jsonl')
    with open(extracted_filename, "w", encoding="utf-8") as f:
        for line in new_json_lines:
            json_string = json.dumps(line)
            f.write(json_string + '\n')
    
    print(f"Extracted responses saved to: {extracted_filename}")
    return extracted_filename


def run_llm_evaluation(code_sum_data: List[Dict], judge_model: str = "gpt-4o-2024-11-20", max_k: int = 1, num_workers: int = None) -> List[Dict]:
    """Run LLM evaluation on code summarization data with parallel processing."""
    print(f"Running LLM evaluation on {len(code_sum_data)} items (max_k={max_k})")
    print(f"Using LLM judge: {judge_model}")
    
    # Get LLM judge
    llm_judge = MODELS.get(judge_model)
    if not llm_judge:
        print(f"Warning: Could not load LLM judge model {judge_model}, using default scores")
        llm_judge = None
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(4, get_optimal_worker_count(len(code_sum_data)))
    else:
        num_workers = min(num_workers, 4)  # Cap at 4 to prevent API rate limiting
    
    print(f"Using {num_workers} workers for evaluation")
    
    # Prepare arguments for each item
    item_args = [(data, llm_judge, judge_model) for data in code_sum_data]
    
    # Process items in parallel
    evaluated_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_process_code_summarization_item, item_args),
            total=len(item_args),
            desc="Evaluating code summarization items"
        ))
        evaluated_data = [result for result in results if result is not None]
    
    print(f"Completed evaluation: {len(evaluated_data)} items processed")
    return evaluated_data


def evaluate_code_summarization_file(filename: str, max_k: int = 1, judge_model: str = "gpt-4o-2024-11-20", 
                                    auto_parse: bool = None, extract_only: bool = False, num_workers: int = None, 
                                    use_batch_evaluation: bool = True) -> str:
    """
    Hybrid code summarization evaluation supporting both predictions and parsed files.
    
    Args:
        filename: Path to input file (predictions or parsed)
        max_k: Maximum number of responses to evaluate per item
        judge_model: LLM judge model name for evaluation
        auto_parse: Whether to auto-parse predictions (None=auto-detect, True=force, False=expect parsed)
        extract_only: If True, only extract/parse without evaluation
        num_workers: Number of parallel workers
        use_batch_evaluation: Use old TREAT batch evaluation pattern (recommended)
        
    Returns:
        Path to the evaluation file created
    """
    print(f"Processing {filename}")
    print(f"Using judge model: {judge_model}")
    print(f"Batch evaluation mode: {use_batch_evaluation}")
    
    # Determine dataset directory from filename
    # Expected path: results/code_summarization/<dataset>/{predictions,parsed}/<model>.jsonl
    if "/code_summarization/" in filename:
        # Extract dataset directory
        parts = filename.split("/code_summarization/")
        if len(parts) == 2:
            dataset_path = parts[1]
            dataset_name = dataset_path.split("/")[0]
            dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        else:
            # Fallback: determine from data
            dataset_dir = None
    else:
        dataset_dir = None
    
    # Use batch evaluation if we can determine dataset directory and it's enabled
    if use_batch_evaluation and dataset_dir and os.path.exists(dataset_dir):
        print(f"Using batch evaluation for dataset: {dataset_dir}")
        
        # Check if this is an extract-only request
        if extract_only:
            # Handle individual file extraction
            print(f"Loading data from: {filename}")
            data = []
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            # Auto-detect file type if not specified
            if auto_parse is None:
                file_type = _detect_file_type(filename, data)
                auto_parse = (file_type == "predictions")
            
            # Convert predictions to parsed format if needed
            if auto_parse:
                data = _convert_predictions_to_parsed(data)
            
            # Save parsed file
            parsed_dir = filename.replace("/predictions/", "/parsed/")
            parsed_dir = os.path.dirname(parsed_dir)
            os.makedirs(parsed_dir, exist_ok=True)
            parsed_file = os.path.join(parsed_dir, os.path.basename(filename))
            
            with open(parsed_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Extraction complete: {parsed_file}")
            return parsed_file
        else:
            # Use batch evaluation
            cache_file = batch_evaluate_dataset(dataset_dir, judge_model)
            
            # Return the evaluation file for this specific model
            model_name = os.path.basename(filename).replace('.jsonl', '')
            evaluation_file = os.path.join(dataset_dir, "evaluations", f"{model_name}.jsonl")
            
            if os.path.exists(evaluation_file):
                print(f"Pipeline completed: {evaluation_file}")
                return evaluation_file
            else:
                print(f"Warning: Evaluation file not created for {model_name}")
                return cache_file
    
    # Fallback to individual evaluation (original approach)
    print("Using individual evaluation mode (fallback)")
    
    print(f"Loading data from: {filename}")
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} items")
    
    # Auto-detect file type if not specified
    if auto_parse is None:
        file_type = _detect_file_type(filename, data)
        auto_parse = (file_type == "predictions")
        print(f"Auto-detected file type: {file_type}")
    
    # Convert predictions to parsed format if needed
    if auto_parse:
        data = _convert_predictions_to_parsed(data)
    
    # Extract-only mode: save parsed file and return
    if extract_only:
        parsed_dir = filename.replace("/predictions/", "/parsed/").replace(".jsonl", "")
        parsed_dir = os.path.dirname(parsed_dir)
        os.makedirs(parsed_dir, exist_ok=True)
        parsed_file = os.path.join(parsed_dir, os.path.basename(filename))
        
        with open(parsed_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Extraction complete: {parsed_file}")
        return parsed_file
    
    # Determine dataset and create output paths
    dataset = "general"
    if data and 'dataset' in data[0]:
        dataset = data[0]['dataset']
    elif "github" in filename.lower():
        dataset = "github"
    
    evaluation_dir = os.path.join(RESULTS_DIR, dataset, "evaluations")
    os.makedirs(evaluation_dir, exist_ok=True)
    evaluation_file = os.path.join(evaluation_dir, os.path.basename(filename))
    
    # Load existing evaluations for breakpoint resuming
    existing_evaluations = _load_existing_evaluations(evaluation_file, judge_model)
    print(f"Found {len(existing_evaluations)} items already evaluated with {judge_model}")
    
    # Filter out already evaluated items
    new_items = []
    for item in data:
        # Generate ref_key if missing
        if 'ref_key' not in item:
            extractor = CodeSummarizationExtractor()
            item['ref_key'] = extractor.make_unique_key(item)
        
        ref_key = item['ref_key']
        if ref_key not in existing_evaluations:
            new_items.append(item)
    
    print(f"Processing {len(new_items)} new items")
    
    if new_items:
        # Run evaluation on new items
        evaluated_items = run_llm_evaluation(new_items, judge_model, max_k, num_workers)
        
        # Combine with existing evaluations
        all_evaluations = list(existing_evaluations.values()) + evaluated_items
        
        # Write all evaluations to file
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            for item in all_evaluations:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Code summarization evaluation completed: {len(evaluated_items)} new items evaluated")
    else:
        print("All items already evaluated!")
    
    print(f"Total items in evaluation file: {len(data)}")
    print(f"Results saved to: {evaluation_file}")
    print(f"Pipeline completed: {evaluation_file}")
    
    return evaluation_file


def process_pipeline(filename: str, k_list: List[int] = [1], extract_only: bool = False, 
                    evaluate_datasets: List[str] = None, num_workers: int = None) -> str:
    """Legacy processing pipeline - redirects to hybrid evaluator."""
    max_k = max(k_list) if k_list else 1
    return evaluate_code_summarization_file(
        filename, 
        max_k=max_k, 
        extract_only=extract_only, 
        num_workers=num_workers
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python code_summarization_evaluator.py <prediction_file> [k_values...] [--extract-only]")
        print("Example: python code_summarization_evaluator.py save/code_summarization_model.jsonl 1 5")
        sys.exit(1)
    
    filename = sys.argv[1]
    extract_only = "--extract-only" in sys.argv
    
    # Parse k values
    k_list = []
    for arg in sys.argv[2:]:
        if arg.isdigit():
            k_list.append(int(arg))
    
    if not k_list:
        k_list = [1]  # Default k value
    
    print(f"Processing {filename} with k={k_list}")
    if extract_only:
        print("Extract only mode - no LLM evaluation")
    
    process_pipeline(filename, k_list=k_list, extract_only=extract_only)
    print(f"Processing complete")