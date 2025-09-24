import unicodedata
import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
def normalize_unicode_characters(code: str) -> str:
    """Normalize problematic Unicode characters in code."""
    if not code:
        return code
    
    # Dictionary of problematic Unicode characters to ASCII replacements
    unicode_replacements = {
        # Smart quotes
        '\u2018': "'",  # '
        '\u2019': "'",  # '
        '\u201c': '"',  # "
        '\u201d': '"',  # "
        
        # Dashes and hyphens
        '\u2013': '-',  # –
        '\u2014': '-',  # —
        '\u2212': '-',  # −
        
        # Other problematic characters
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero-width space
        '\u200c': '',   # Zero-width non-joiner
        '\u200d': '',   # Zero-width joiner
        '\ufeff': '',   # Byte order mark
        
        # Mathematical symbols that might be misused
        '\u2260': '!=', # ≠
        '\u2264': '<=', # ≤
        '\u2265': '>=', # ≥
    }
    
    # Apply replacements
    cleaned_code = code
    for unicode_char, replacement in unicode_replacements.items():
        cleaned_code = cleaned_code.replace(unicode_char, replacement)
    
    # Normalize any remaining Unicode to closest ASCII equivalent
    cleaned_code = unicodedata.normalize('NFKD', cleaned_code)
    
    return cleaned_code


def get_optimal_worker_count(data_size: int, max_workers: int = None, cap_workers: int = None) -> int:
    """
    Determine optimal number of workers based on data size and system specs.
    
    Args:
        data_size: Number of items to process
        max_workers: Maximum workers to use (default: CPU count)
        cap_workers: Hard cap on workers (useful for resource-intensive tasks)
        
    Returns:
        Optimal number of workers
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Apply cap if specified
    if cap_workers is not None:
        max_workers = min(max_workers, cap_workers)
    
    # Don't use more workers than data items
    optimal = min(max_workers, data_size, mp.cpu_count())
    # Ensure at least 1 worker
    return max(1, optimal)


def split_prediction_file_by_dataset(filename: str, save_dir: str) -> Dict[str, str]:
    """
    Split a prediction file by dataset for separate evaluation.
    
    Args:
        filename: Path to the prediction file
        save_dir: Directory to save split files
        
    Returns:
        Dict mapping dataset names to split file paths
    """
    dataset_files = defaultdict(list)
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                dataset = data.get('dataset', 'unknown').lower()
                dataset_files[dataset].append(data)
            except json.JSONDecodeError:
                continue
    
    # Write split files
    split_files = {}
    for dataset, items in dataset_files.items():
        if items:  # Only create file if we have data
            split_filename = os.path.join(save_dir, f"split_{dataset}.jsonl")
            with open(split_filename, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            split_files[dataset] = split_filename
            
    return split_files


def save_evaluation_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save evaluation results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the results
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result:  # Skip None results
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def load_prediction_file(filename: str) -> List[Dict[str, Any]]:
    """
    Load predictions from a JSONL file.
    
    Args:
        filename: Path to the prediction file
        
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return predictions


def create_results_directory(task_name: str, dataset_name: str = None, base_dir: str = None) -> Path:
    """
    Create and return the results directory path for a task and dataset.
    
    Args:
        task_name: Name of the task (e.g., 'code_generation')
        dataset_name: Name of the dataset (optional)
        base_dir: Base results directory (default: project_root/results)
        
    Returns:
        Path object for the results directory
    """
    if base_dir is None:
        # Assume we're in evaluators/utils, so go up two levels
        project_root = Path(__file__).parent.parent.parent
        base_dir = project_root / "results"
    else:
        base_dir = Path(base_dir)
    
    if dataset_name:
        results_dir = base_dir / task_name / dataset_name
    else:
        results_dir = base_dir / task_name
    
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def format_evaluation_summary(task_name: str, processed_files: List[str], results_dir: str) -> None:
    """
    Print a formatted summary of evaluation completion.
    
    Args:
        task_name: Name of the evaluation task
        processed_files: List of processed file paths or names
        results_dir: Directory where results were saved
    """
    print(f"\n{'='*50}")
    print(f"{task_name.upper().replace('_', ' ')} EVALUATION COMPLETE")
    print(f"{'='*50}")
    
    if processed_files:
        print(f"Processed files:")
        for file_path in processed_files:
            print(f"  - {os.path.basename(file_path)}")
    else:
        print("No files were processed")
    
    print(f"Results saved to: {results_dir}")
    print("Evaluation metrics include:")
    print("  - Pass@k scores (k=1,5,10)")  
    print("  - Success rates")
    print("  - Detailed execution results")

def remove_thinking_tag(response: str) -> str:
    """
    Extract content after <thinking> and </thinking> tags from the response string.
    Args:
        response (str): Input string that contains thinking tags
    Returns:
        str: final response after removing the thinking
    """
    try:
        response = response.split("</think>")[-1]
    except:
        pass
    return response

def extract_evaluation_json(string: str) -> Optional[str]:
    """Extract JSON from markdown code blocks."""
    pattern = r"```\w+\s*\n*\s*(.*?)\s*\n*\s*```"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    
def parse_response_outputs(outputs: List[str], lang: str) -> List[str]:
    """Parse response outputs to extract comments."""
    extracted_outputs = []
    
    for output in outputs:
        # 1. Look for ```{language}<content>``` pattern
        match = re.search(r'```\w+\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            content = match.group(1).strip()
            
            if lang == 'python':
                if content.startswith("#"):
                    content = content[1:].strip()
                pattern = r'#\s*(.*)|"""\n?\s*([\s\S]*?)\s*\n?"""'
                matches = re.findall(pattern, content)
                content = [m[0] if m[0] else m[1] for m in matches]
                content = ''.join(content)
            
            extracted_outputs.append(content if content.endswith(".") else content + ".")
            continue
        
        # 2. Look for ```<content>``` pattern  
        match = re.search(r'```\n(.*?)```', output, re.DOTALL)
        if match:
            content = match.group(1).strip()
            extracted_outputs.append(content if content.endswith(".") else content + ".")
            continue
        
        # 3. Look for colon ":" to first full stop "."
        if output.find(":") != -1 and output.find(".") != -1 and output.find("::") == -1:
            first_colon = output.find(":")
            first_full_stop = output.find(".")
            content = output[first_colon+1:first_full_stop+1].strip()
            extracted_outputs.append(content)
            continue
        
        # 4. Parse the whole sentence as output
        content = output.strip()
        if lang == 'python' and content.startswith("#"):
            content = content[1:].strip()
        
        extracted_outputs.append(content if content.endswith(".") else content + ".")

    return extracted_outputs


def escape_control_characters(json_str: str) -> str:
    """
    Escape ASCII control characters in JSON strings using regex.
    
    Args:
        json_str: JSON string that may contain unescaped control characters
        
    Returns:
        JSON string with control characters properly escaped
    """
    def replace(match):
        ch = match.group(0)
        return '\\u%04x' % ord(ch)
    
    # (?<!\\) ensures we don't match already escaped characters
    return re.sub(r'(?<!\\)([\x00-\x1F])', replace, json_str)


def extract_comment_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from code review response that may be wrapped in Markdown.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Parsed JSON object or None if extraction fails
    """
    import hjson
    
    # Try to capture JSON content inside Markdown code fences
    pattern_fenced = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern_fenced, response, re.DOTALL)
    
    # Fallback: If no fenced block, try simpler JSON pattern
    if not match:
        pattern_simple = r'(\{\s*"comments":\s*".*?"\s*\})'
        match = re.search(pattern_simple, response, re.DOTALL)

    if not match:
        # Try to find any JSON-like structure
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end+1]
            json_str = escape_control_characters(json_str)
            json_str = json_str.replace("\\u000a", "")
            json_str = re.sub(r'(?<!\\)\n', r'\\n', json_str)
            try:
                return hjson.loads(json_str)
            except Exception:
                return None
        else:
            return None

    if match:
        json_str = match.group(1)
        json_str = escape_control_characters(json_str)
        json_str = json_str.replace("\\u000a", "")
        json_str = re.sub(r'(?<!\\)\n', r'\\n', json_str)
        try:
            return hjson.loads(json_str)
        except Exception:
            return None
    
    return None


def parse_code_review_outputs(outputs: List[str]) -> List[str]:
    """
    Parse code review response outputs to extract review comments.
    
    Args:
        outputs: List of raw response strings from LLM
        
    Returns:
        List of extracted review comments
    """
    extracted_outputs = []
    
    for output in outputs:
        if not output or output == "Exceeds Context Length":
            extracted_outputs.append("")
            continue
            
        try:
            # Try to extract JSON response first
            parsed_response = extract_comment_response(output)
            if parsed_response and 'comments' in parsed_response:
                comment = parsed_response['comments']
                if isinstance(comment, list):
                    comment = '\n'.join(str(c) for c in comment)
                extracted_outputs.append(str(comment))
            else:
                # Fallback: use the raw output
                extracted_outputs.append(output.strip())
                
        except Exception:
            # If all else fails, use raw output
            extracted_outputs.append(output.strip())
    
    return extracted_outputs