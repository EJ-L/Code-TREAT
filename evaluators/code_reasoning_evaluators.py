from evaluators.utils.code_exec_java_executor import JavaExecutor
from evaluators.utils.code_exec_python_executor import PythonExecutor
import json
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Tuple, Any, Optional
import argparse
import logging
from ast import literal_eval
import re
import numpy as np
from tests.code_execution.evaluation.utils_general import pass_at_k, extract_prediction, parse_java_md, parse_python_md, handle_input_prediction_response, handle_output_prediction_response
from tests.code_execution.evaluation.utils import find_output_prediction_files, find_input_prediction_files
from tqdm import tqdm
import os

JAVA_CLS_NAME_DICT = {
    6715: "Kaprekar",
    6024: "AngryProf", 
    19310: "Vic",
    702766: "Compute", 
    702678: "Solve", 
    700428: "GfG", 
    710277: "solver", 
    700619: "Tree"
}
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

# 搜索輸出 / 輸入 prediction 的起始位置
_HEAD_RE = re.compile(
    r'\{\s*"(input|output)[ _]prediction"\s*:\s*',
    re.I  # ignore-case
)

def grab_prediction(text: str) -> Optional[Tuple[str, str]]:
    """
    抓出第一段 {"input/output_prediction": …}，
    回傳 (完整片段, value 字串)；找不到就 None。
    """
    m = _HEAD_RE.search(text)
    if not m:
        return None

    start = m.start()   # '{'
    i     = m.end()     # 冒號後第一字元
    depth = 1           # 已看到 1 個 {
    in_str, esc = False, False

    while i < len(text):
        ch = text[i]

        if in_str:                    # 在字串裏
            if esc:
                esc = False           # 上一個是跳脫符號，直接略過
            elif ch == '\\':
                esc = True            # 開始跳脫
            elif ch == '"':
                in_str = False        # 字串結束
        else:                         # 不在字串裏
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:        # 找到配對的右括號
                    fragment   = text[start : i + 1]
                    value_part = text[m.end() : i]   # 冒號後到倒數 `}` 前
                    return fragment, value_part.strip()
        i += 1

    return None   # 若跑到結尾 depth 仍 >0，表示 json 不完整

def parse_json_md(markdown_text):
    """
    Extracts Java code enclosed in ```java ... ``` blocks from a markdown string.
    Returns a list of code block strings.
    """
    pattern = r"```json\n(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    if matches and len(matches) == 1:
        return matches[0], True
    else:
        return None

def supplement_inputs_in_py(input_line: str, masked_code: str):
    if not masked_code.startswith("inputs") and ('assert Solution().f(*inputs)' in masked_code or 'assert Solution().f(inputs)' in masked_code or 'assert f(*inputs)' in masked_code or 'assert f(inputs)' in masked_code):
        return input_line + '\n' + masked_code
    return masked_code

def parse_response(resp: List[str], masked_query: str, mode: str="input", lang="python"):
    parsed_responses = []
    parsed_case = []
    if mode == "input":          # ---------- input prediction ----------
        key = "input_prediction"
    else:                        # ---------- output prediction ----------
        key = "output_prediction"
        
    if lang == 'java':
        # parse cases: -1: plain string that is probably wrong, 1 -> json format, 2 -> plain text starting with ```java or ```python starting with psvm, 3 -> plain string markdown, 4 -> json format w/o psvm, 5 -> ```java but not starting with psvm code, probably entire code`
        for r in resp:
            r = remove_thinking_tag(r)
            if "```" in r and ("```java" in r or "```python" in r):
                success = False
                _tuple = parse_java_md(r)
                if _tuple is not None:
                    content, success = _tuple
                if success:
                    parsed_responses.append(content)
                    parsed_case.append(2 if content.lstrip().startswith("public static void main") else 5)
                    continue
            try:
                # grab_prediction 先嘗試 {"output_prediction": …}
                # parse_json_md 處理 ```json ...``` 或破格式
                json_r = grab_prediction(r) or parse_json_md(r)
                if json_r is not None:
                    r, _ = json_r
            except Exception:
                pass
            
            try:
                try:
                    r = json.loads(r)
                except Exception:
                    r = literal_eval(r)
            except Exception:
                pass
            
            if isinstance(r, dict) and key in r:
                pred = r[key]
                pred_str = str(pred).strip()
                if pred_str.startswith("public static void main"):
                    parsed_responses.append(pred_str)
                    parsed_case.append(1)
                else:
                    parsed_responses.append(masked_query.replace("??", pred_str))
                    parsed_case.append(4)
            elif isinstance(r, str):
                rs = r.strip()
                if rs.startswith("public static void main"):
                    parsed_responses.append(rs)
                    parsed_case.append(2)
                else:
                    parsed_responses.append(rs)
                    parsed_case.append(3)
            else:
                # 解析失敗，原樣丟回，可視需求決定是否 append
                parsed_responses.append(r)
                parsed_case.append(-1)
    elif lang == 'python':
        # 1 -> json + 
        for r in resp:        
            r = remove_thinking_tag(r)
            if r.strip() == "":      
                parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                parsed_case.append(-2)
                continue
            from_json = False
            successful_response = None
            parsed_response = extract_prediction(r, True)
            # if parsed_response, further handle the json
            if parsed_response:
                successful_response = parsed_response[0][key]
                from_json = True
            else:
                _tuple = parse_python_md(r)
                if _tuple is not None:
                    content, success = _tuple
                    if success:
                        successful_response = content[0]
            if mode == 'input':
                print("=====")
                print(r)
                if successful_response:
                    final_parse, idx = handle_input_prediction_response(successful_response, from_json)
                    parsed_responses.append(masked_query.replace("??", final_parse, 1))
                    parsed_case.append(idx)
                    print("+++++")
                    print(parsed_responses[0])
                else:
                    parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                    parsed_case.append(-2)
            if mode == 'output':
                if successful_response:
                    print(successful_response)
                    final_parse, idx = handle_output_prediction_response(successful_response, from_json)
                    parsed_responses.append(masked_query.replace("??", final_parse, 1))
                    parsed_case.append(idx)
                    print(parsed_responses[0])
                else:
                    parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                    parsed_case.append(-2)
    else:
        raise ValueError('Language apart from python and java are not supported')

    return parsed_responses, parsed_case

def parse_error_details(error_output: str) -> tuple:
    """
    Parse error output to extract error type and code for better reporting.
    
    Args:
        error_output: Error message from execution
        
    Returns:
        Tuple of (error_type, error_code)
    """
    if not error_output:
        return "unknown", None
    
    error_output_lower = error_output.lower()
    
    # Compilation errors
    if "compilation failed" in error_output_lower:
        error_type = "compilation_error"
        
        # Extract specific compilation error codes
        if "class, interface, enum, or record expected" in error_output_lower:
            error_code = "SYNTAX_CLASS_EXPECTED"
        elif "cannot find symbol" in error_output_lower:
            error_code = "SYMBOL_NOT_FOUND"
        elif "illegal start of expression" in error_output_lower:
            error_code = "ILLEGAL_EXPRESSION_START"
        elif "';' expected" in error_output_lower:
            error_code = "SEMICOLON_EXPECTED"
        elif "')' expected" in error_output_lower:
            error_code = "PARENTHESIS_EXPECTED"
        elif "'{' expected" in error_output_lower:
            error_code = "BRACE_EXPECTED"
        elif "incompatible types" in error_output_lower:
            error_code = "TYPE_MISMATCH"
        elif "unreachable statement" in error_output_lower:
            error_code = "UNREACHABLE_STATEMENT"
        elif "duplicate" in error_output_lower:
            error_code = "DUPLICATE_DECLARATION"
        else:
            error_code = "OTHER_COMPILATION_ERROR"
    
    # Runtime errors
    elif "runtime error" in error_output_lower:
        error_type = "runtime_error"
        
        if "nullpointerexception" in error_output_lower:
            error_code = "NULL_POINTER_EXCEPTION"
        elif "arrayindexoutofboundsexception" in error_output_lower:
            error_code = "ARRAY_INDEX_OUT_OF_BOUNDS"
        elif "classcastexception" in error_output_lower:
            error_code = "CLASS_CAST_EXCEPTION"
        elif "arithmeticexception" in error_output_lower:
            error_code = "ARITHMETIC_EXCEPTION"
        elif "stackoverflowerror" in error_output_lower:
            error_code = "STACK_OVERFLOW"
        elif "outofmemoryerror" in error_output_lower:
            error_code = "OUT_OF_MEMORY"
        elif "assertionerror" in error_output_lower:
            error_code = "ASSERTION_FAILED"
        else:
            error_code = "OTHER_RUNTIME_ERROR"
    
    # Timeout errors
    elif "timeout" in error_output_lower:
        error_type = "timeout_error"
        error_code = "EXECUTION_TIMEOUT"
    
    # Execution errors (general)
    elif "execution error" in error_output_lower:
        error_type = "execution_error"
        error_code = "GENERAL_EXECUTION_ERROR"
    
    # Python-specific errors
    elif "syntaxerror" in error_output_lower:
        error_type = "syntax_error"
        error_code = "PYTHON_SYNTAX_ERROR"
    elif "indentationerror" in error_output_lower:
        error_type = "syntax_error"
        error_code = "PYTHON_INDENTATION_ERROR"
    elif "nameerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_NAME_ERROR"
    elif "typeerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_TYPE_ERROR"
    elif "valueerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_VALUE_ERROR"
    elif "indexerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_INDEX_ERROR"
    elif "keyerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_KEY_ERROR"
    elif "attributeerror" in error_output_lower:
        error_type = "runtime_error"
        error_code = "PYTHON_ATTRIBUTE_ERROR"
    
    else:
        error_type = "unknown"
        error_code = "UNCLASSIFIED_ERROR"
    
    return error_type, error_code

def parse_generations(filename: str, directory: str) -> List[Dict[str, Any]]:
    """Parse generation file and return the full path of parsed jsonl file"""
    # Ensure target directory exists
    os.makedirs(directory, exist_ok=True)
    basename = os.path.basename(filename).replace(".jsonl", "_parsed.jsonl")
    out_path = os.path.abspath(os.path.join(directory, basename))
    with open(filename, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as f2:
        for line_num, line in enumerate(f.readlines(), 1):
            try:     
                print(line_num)

                json_line = json.loads(line.strip())

                if all(key in json_line for key in ['code/function', 'lang', 'id']):
                    try:
                        masked_query = literal_eval(json_line['test_case_metadata'])[json_line['task']]
                    except:
                        masked_query = json_line['test_case_metadata'][json_line['task']]
                    
                    # parse the response generated by LLMs, task will be either input_prediction/output_predidction
                    if 'parsed_response' in json_line:
                        pass
                    else:
                        print("HERE")
                        json_line['parsed_response'], json_line['parsed_case'] = parse_response(json_line['response'], masked_query, json_line['task'].split("_")[0], json_line['lang'])
                        print(json_line['model_name'])
                        # json_line['parsed_response'] = [
                        #     base64.b64encode(pickle.dumps(r)).decode("ascii") for r in json_line['parsed_response']
                        # ]
                    f2.write(json.dumps(json_line) + '\n')
            except json.JSONDecodeError as e:
                print(json_line['response'])
                raise Exception(e)
    return out_path

def evaluate_single_generation(args_tuple: Tuple[Dict[str, Any], List[int]]) -> Dict[str, Any]:
    """Evaluate a single generation - designed for multiprocessing"""
    test_data, k_values = args_tuple
    try:
        test_masked_code = test_data['code/function']
        lang = test_data['lang']
        problem_id = test_data['id']
        test_generations = test_data.get('parsed_response', [])
        
        test_case_idx = test_data['test_case_idx']
        # Get class name from problem ID
        cls_name = JAVA_CLS_NAME_DICT.get(problem_id, "Solution")
        try:
            input_line = test_data['test_case_metadata']['input']
        except:
            try:
                input_line = literal_eval(test_data['test_case_metadata'])['input']
            except:
                input_line = json.loads(test_data['test_case_metadata'])['input']
        # Create executor based on language
        if lang.lower() == 'java':
            executor = JavaExecutor(test_masked_code, cls_name) 
        elif lang.lower() == 'python':
            executor = PythonExecutor(test_masked_code)
        else:
            raise ValueError("Language not suporrted")
        # Execute test cases
        results = []
        
        for _, gen in enumerate(test_generations):
            if lang.lower() == 'python':
                gen = supplement_inputs_in_py(input_line, gen)
            exec_code, success, output  = executor.execute_test_case(gen)
            # Parse error details for better reporting
            error_type = "none" if success else "unknown"
            error_code = None
            
            if not success:
                error_type, error_code = parse_error_details(output)
            
            results.append({
                'prompt_category': test_data['prompt_category'],
                'prompt_id': test_data['prompt_id'],
                'test_case_id': test_case_idx,
                'success': success,
                'output': output,
                'error_type': error_type,
                'error_code': error_code,
                'exec_code': exec_code
            })
        
        # Calculate pass@k scores
        success_results = [r['success'] for r in results]
        pass_at_k_scores = {}
        for k in k_values:
            if k <= len(success_results):
                n = len(success_results)
                c = sum(success_results)
                pass_at_k_scores[f'pass@{k}'] = pass_at_k(n, c, k)
        
        return {
            'problem_id': problem_id,
            'lang': lang,
            'success_count': sum(1 for r in results if r['success']),
            'total_tests': len(results),
            'pass_at_k': pass_at_k_scores,
            'results': results
        }
        
    except Exception as e:
        return {
            'problem_id': test_data.get('id', 'unknown'),
            'lang': test_data.get('lang', 'unknown'),
            'success_count': 0,
            'total_tests': 0,
            'pass_at_k': {},
            'error': str(e),
            'results': []
        }

def evaluate_generations(filename: str, max_workers: int = 4, output_file: str = None, k_values: List[int] = [1, 5, 10]) -> List[Dict[str, Any]]:
    """Evaluate all generations using ProcessPoolExecutor"""
    generations = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            json_line = json.loads(line)
            generations.append(json_line)

    
    # Parse generations
    # generations = parse_generations(filename)
    
    if not generations:
        return []
    
    # Detect if this is primarily Java code
    java_heavy = any('java' in gen.get('lang', '').lower() for gen in generations[:10])
    
    # Use ThreadPoolExecutor for Java (better for I/O bound compilation) and ProcessPoolExecutor for Python
    ExecutorClass = ThreadPoolExecutor if java_heavy else ProcessPoolExecutor
    
    # Evaluate using selected executor
    results = []
    start_time = time.time()
    
    with ExecutorClass(max_workers=max_workers) as executor:
        # Submit all tasks with k_values
        future_to_data = {executor.submit(evaluate_single_generation, (gen, k_values)): gen for gen in generations}
        
        # Collect results as they complete with progress bar
        with tqdm(total=len(generations), desc="Evaluating generations", unit="problem") as pbar:
            for future in as_completed(future_to_data):
                gen_data = future_to_data[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar with current problem info
                    success_rate = f"{result['success_count']}/{result['total_tests']}"
                    pbar.set_postfix({
                        'Current': f"P{result['problem_id']}",
                        'Success': success_rate,
                        'Lang': result['lang']
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    results.append({
                        'problem_id': gen_data.get('id', 'unknown'),
                        'lang': gen_data.get('lang', 'unknown'),
                        'success_count': 0,
                        'total_tests': 0,
                        'pass_at_k': {},
                        'error': str(e),
                        'results': []
                    })
                    pbar.update(1)
    
    hash_table = {}          
    pass_at_1 = []
    for result in results:
        r = result['results']
        key = f"{result['problem_id']},{result['lang']},{str(r[0]['prompt_category'])},{str(r[0]['prompt_id'])},{str(r[0]['test_case_id'])}"
        hash_table[key] = (result['pass_at_k'], r)        
        pass_at_1.append(result['pass_at_k']['pass@1'])
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    with open(str(filename).replace(".jsonl", "_evaluated.jsonl"), "w", encoding="utf-8") as f:
        for line in generations:
            key = f"{line['id']},{line['lang']},{str(line['prompt_category'])},{str(line['prompt_id'])},{str(line['test_case_idx'])}"
            line['metrics'] = hash_table[key][0]
            line['results'] = hash_table[key][1]
            f.write(json.dumps(line) + '\n')        
    return sum(pass_at_1) / len(pass_at_1)

def print_summary(results: List[Dict[str, Any]], k_values: List[int] = [1, 5, 10]):
    """Print evaluation summary"""
    total_problems = len(results)
    total_success = sum(r['success_count'] for r in results)
    total_tests = sum(r['total_tests'] for r in results)
    
    java_results = [r for r in results if r['lang'].lower() == 'java']
    python_results = [r for r in results if r['lang'].lower() == 'python']
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Problems: {total_problems}")
    print(f"Total Test Cases: {total_tests}")
    print(f"Total Successful: {total_success}")
    print(f"Overall Success Rate: {total_success/total_tests*100:.2f}%" if total_tests > 0 else "No tests executed")
    
    # Calculate overall pass@k scores
    all_pass_at_k = {}
    for k in k_values:
        valid_results = [r for r in results if f'pass@{k}' in r.get('pass_at_k', {})]
        if valid_results:
            avg_pass_k = np.mean([r['pass_at_k'][f'pass@{k}'] for r in valid_results])
            all_pass_at_k[f'pass@{k}'] = avg_pass_k
    
    print(f"\nOverall Pass@K Scores:")
    for k, score in all_pass_at_k.items():
        print(f"  {k}: {score:.4f}")
    
    if java_results:
        java_success = sum(r['success_count'] for r in java_results)
        java_total = sum(r['total_tests'] for r in java_results)
        print(f"\nJava: {len(java_results)} problems, {java_success}/{java_total} tests passed "
              f"({java_success/java_total*100:.2f}%)" if java_total > 0 else "")
        
        # Java pass@k scores
        java_pass_at_k = {}
        for k in k_values:
            valid_java = [r for r in java_results if f'pass@{k}' in r.get('pass_at_k', {})]
            if valid_java:
                java_pass_at_k[f'pass@{k}'] = np.mean([r['pass_at_k'][f'pass@{k}'] for r in valid_java])
        
        for k, score in java_pass_at_k.items():
            print(f"  Java {k}: {score:.4f}")
    
    if python_results:
        python_success = sum(r['success_count'] for r in python_results)
        python_total = sum(r['total_tests'] for r in python_results)
        print(f"\nPython: {len(python_results)} problems, {python_success}/{python_total} tests passed "
              f"({python_success/python_total*100:.2f}%)" if python_total > 0 else "")
        
        # Python pass@k scores
        python_pass_at_k = {}
        for k in k_values:
            valid_python = [r for r in python_results if f'pass@{k}' in r.get('pass_at_k', {})]
            if valid_python:
                python_pass_at_k[f'pass@{k}'] = np.mean([r['pass_at_k'][f'pass@{k}'] for r in valid_python])
        
        for k, score in python_pass_at_k.items():
            print(f"  Python {k}: {score:.4f}")
    
    # Show error statistics
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nProcessing errors encountered: {len(error_results)} problems")
        for r in error_results[:5]:  # Show first 5 errors
            print(f"  - Problem {r['problem_id']}: {r['error']}")
    
    # Show detailed error type statistics
    print(f"\nError Type Statistics:")
    error_type_counts = {}
    error_code_counts = {}
    
    for result in results:
        for test_result in result.get('results', []):
            if not test_result['success']:
                error_type = test_result.get('error_type', 'unknown')
                error_code = test_result.get('error_code', 'UNKNOWN')
                
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                error_code_counts[error_code] = error_code_counts.get(error_code, 0) + 1
    
    if error_type_counts:
        print(f"  Error Types:")
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {error_type}: {count}")
        
        print(f"  Top Error Codes:")
        for error_code, count in sorted(error_code_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {error_code}: {count}")
    else:
        print(f"  No execution errors found!")

def main():
    parser = argparse.ArgumentParser(description='Evaluate code generations using ProcessPoolExecutor')
    parser.add_argument('input_file', help='Input JSONL file with generations')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes (default: 4)')
    parser.add_argument('--output', help='Output file for results (JSONL format)')
    parser.add_argument('--log-file', help='Log file path (logs to console if not specified)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 5, 10], 
                        help='List of k values for pass@k calculation (default: 1 5 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, args.verbose)
    
    # Run evaluation
    results = evaluate_generations(args.input_file, args.workers, args.output, args.k_values)
    
    # Print summary
    print_summary(results, args.k_values)

if __name__ == '__main__':
    # main()
    # print(grab_prediction("{\"output_prediction\": \"public static void main(String[] args) {\n    step[] steps = new step[2];\n    steps[0] = new step();\n    steps[0].departure = 10;\n    steps[0].travelTime = 4;\n    steps[0].distance = 0;\n    steps[0].pickedUp = 0;\n    steps[0].dropped = 0;\n    steps[0].carried = 0;\n    steps[1] = new step();\n    steps[1].departure = 5;\n    steps[1].travelTime = 3;\n    steps[1].distance = 0;\n    steps[1].pickedUp = 0;\n    steps[1].dropped = 0;\n    steps[1].carried = 0;\n    passenger[] passengers = new passenger[2];\n    passengers[0] = new passenger();\n    passengers[0].arrival = 2;\n    passengers[0].start = 0;\n    passengers[0].dest = 1;\n    passengers[1] = new passenger();\n    passengers[1].arrival = 1;\n    passengers[1].start = 0;\n    passengers[1].dest = 1;\n    assert f(steps, passengers) == 12;\n}\"}"))
    # evaluate_generations("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/save/CodeExecutionParsedV2/output_prediction_Claude-3.5-Haiku-20241022_java_fixed.jsonl", 10, "inspect.jsonl")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "save")
    CODEEXEC_DIR = os.path.join(SAVE_DIR, "CodeExecution")
    PARSED_DIR = os.path.join(SAVE_DIR, "CodeExecutionParsed")
    PARSED_FIXED_DIR = os.path.join(SAVE_DIR, "CodeExecutionParsedV2")
    # filename = "/Users/ericjohnli/Downloads/RA_ARISE/TREAT/save/output_prediction_GPT-5_python.jsonl"
    # parse_generations(filename, PARSED_DIR)
    # files = find_output_prediction_files(CODEEXEC_DIR, '', 'python')
    # for file in files:
    #     parse_generations(file, PARSED_DIR)
    files = find_output_prediction_files(PARSED_FIXED_DIR, mode='fix', lang='python')
    # files = find_input_prediction_files(PARSED_FIXED_DIR, mode='fix', lang='python')
    # files = ["/Users/ericjohnli/Downloads/RA_ARISE/TREAT/save/CodeExecutionParsedV2/output_prediction_GPT-5_python_fixed.jsonl"]
    # print(files)
    for file in files:
        print(file.name)
        if 'GPT-5' in file.name:
            continue
    #     # if file.name.find('Claude-Sonnet-4') != -1 or file.name.find('Llama-3.3-70B-Instruct') != -1:
    #     #     print("SKIPPED")
    #     #     continue
        # Use optimized worker count: more workers for Java with ThreadPoolExecutor
        workers = 8 if 'java' in file.name else 5
        evaluate_generations(file, workers, file.name.replace(".jsonl", "_debug.jsonl"))



