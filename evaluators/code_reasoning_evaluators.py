"""
Code Reasoning Evaluator for standalone evaluation of parsed responses.
Uses the 2-step approach compatible extractors and proper Java/Python executors.
"""

from evaluators.utils.code_exec_java_executor import JavaExecutor
from evaluators.utils.code_exec_python_executor import PythonExecutor
import json
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Tuple, Any
from ast import literal_eval
import numpy as np
from tqdm import tqdm


class CodeReasoningEvaluator:
    """
    Standalone evaluator for code reasoning tasks.
    Evaluates parsed responses using proper Java/Python executors with resume capability.
    """
    
    # Class name mappings for Java problems
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
    
    def __init__(self, max_workers: int = 4, k_values: List[int] = None, max_responses_to_evaluate: int = None):
        """
        Initialize the evaluator
        
        Args:
            max_workers: Number of parallel workers for evaluation
            k_values: List of k values for pass@k calculation (defaults to [1, 5, 10])
            max_responses_to_evaluate: Maximum number of responses from parsed_response to use for evaluation. 
                                     If None, uses all available responses.
        """
        self.max_workers = max_workers
        self.k_values = k_values or [1, 5, 10]
        self.max_responses_to_evaluate = max_responses_to_evaluate
    
    def evaluate_file(self, input_file: str, output_file: str = None, resume: bool = True) -> float:
        """
        Evaluate parsed responses from input file
        
        Args:
            input_file: Path to input JSONL file with parsed responses (in parsed/)
            output_file: Path to output evaluation file (optional, auto-generated in evaluations/)
            resume: Whether to resume from existing evaluation file
            
        Returns:
            Average pass@1 score
        """
        # Get the root directory (where parsed/ folder is located)
        # Assuming input_file is in parsed/file.jsonl
        parsed_dir = os.path.dirname(input_file)
        if os.path.basename(parsed_dir) == 'parsed':
            root_dir = os.path.dirname(parsed_dir)
        else:
            # Fallback: assume input_file directory is the root
            root_dir = parsed_dir
        
        # Setup evaluations directory following framework structure
        evaluations_dir = os.path.join(root_dir, "evaluations")
        os.makedirs(evaluations_dir, exist_ok=True)
        
        # Generate output file path if not provided
        if output_file is None:
            input_basename = os.path.basename(input_file).replace('.jsonl', '')
            output_file = os.path.join(evaluations_dir, f"{input_basename}_evaluation.jsonl")
        
        # Check for resume
        if resume and os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping evaluation.")
            # Load existing results to calculate average pass@1
            return self._calculate_existing_pass_at_1(output_file)
        
        # Load generations
        print(f"Loading generations from {input_file}")
        generations = self._load_generations(input_file)
        
        if not generations:
            print("No generations found to evaluate")
            return 0.0
        
        print(f"Evaluating {len(generations)} generations")
        
        # Evaluate generations
        results = self._evaluate_generations(generations)
        
        # Save results and calculate metrics
        self._save_evaluation_results(results, output_file, generations, input_file)
        
        # Calculate and return average pass@1
        avg_pass_at_1 = self._calculate_average_pass_at_1(results)
        print(f"Average pass@1: {avg_pass_at_1:.4f}")
        
        return avg_pass_at_1
    
    def _load_generations(self, input_file: str) -> List[Dict[str, Any]]:
        """Load generations from input JSONL file"""
        generations = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        json_line = json.loads(line.strip())
                        # Validate required fields
                        if self._validate_generation_line(json_line):
                            generations.append(json_line)
                        else:
                            print(f"Warning: Line {line_num} missing required fields")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            print(f"Input file not found: {input_file}")
        except Exception as e:
            print(f"Error loading generations: {e}")
        
        return generations
    
    def _validate_generation_line(self, json_line: Dict[str, Any]) -> bool:
        """Validate that a generation line has required fields"""
        required_fields = ['code/function', 'lang', 'id', 'parsed_response']
        return all(field in json_line for field in required_fields)
    
    def _evaluate_generations(self, generations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all generations using multiprocessing"""
        # Detect language for optimal executor selection
        java_heavy = any('java' in gen.get('lang', '').lower() for gen in generations[:10])
        ExecutorClass = ThreadPoolExecutor if java_heavy else ProcessPoolExecutor
        
        executor_name = "ThreadPoolExecutor" if java_heavy else "ProcessPoolExecutor"
        print(f"Using {executor_name} with {self.max_workers} workers")
        
        results = []
        start_time = time.time()
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_data = {
                executor.submit(self._evaluate_single_generation, (gen, self.k_values, self.max_responses_to_evaluate)): gen 
                for gen in generations
            }
            
            # Collect results with progress tracking
            with tqdm(total=len(generations), desc="Evaluating generations", unit="problem") as pbar:
                for future in as_completed(future_to_data):
                    gen_data = future_to_data[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress bar
                        success_rate = f"{result['success_count']}/{result['total_tests']}"
                        pbar.set_postfix({
                            'Problem': f"P{result['problem_id']}",
                            'Success': success_rate,
                            'Lang': result['lang']
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error evaluating problem {gen_data.get('id', 'unknown')}: {e}")
                        # Create error result
                        error_result = {
                            'problem_id': gen_data.get('id', 'unknown'),
                            'lang': gen_data.get('lang', 'unknown'),
                            'success_count': 0,
                            'total_tests': 0,
                            'pass_at_k': {},
                            'error': str(e),
                            'results': []
                        }
                        results.append(error_result)
                        pbar.update(1)
        
        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        
        return results
    
    def _evaluate_single_generation(self, args_tuple: Tuple[Dict[str, Any], List[int], Any]) -> Dict[str, Any]:
        """Evaluate a single generation - designed for multiprocessing"""
        test_data, k_values, max_responses_to_evaluate = args_tuple
        
        try:
            # Extract test information
            test_masked_code = test_data['code/function']
            lang = test_data['lang']
            problem_id = test_data['id']
            test_generations = test_data.get('parsed_response', [])
            
            # Limit number of responses if specified
            if max_responses_to_evaluate is not None and len(test_generations) > max_responses_to_evaluate:
                test_generations = test_generations[:max_responses_to_evaluate]
            
            test_case_idx = test_data.get('test_case_idx', 1)
            
            # Get class name for Java
            cls_name = self.JAVA_CLS_NAME_DICT.get(problem_id, "Solution")
            
            # Get input line for Python supplement
            input_line = self._extract_input_line(test_data)
            
            # Create executor based on language
            if lang.lower() == 'java':
                executor = JavaExecutor(test_masked_code, cls_name) 
            elif lang.lower() == 'python':
                executor = PythonExecutor(test_masked_code)
            else:
                raise ValueError(f"Language not supported: {lang}")
            
            # Execute test cases
            results = []
            for gen in test_generations:
                if lang.lower() == 'python':
                    gen = self._supplement_inputs_in_py(input_line, gen)
                
                exec_code, success, output = executor.execute_test_case(gen)
                
                # Parse error details
                error_type = "none" if success else "unknown"
                error_code = None
                if not success:
                    error_type, error_code = self._parse_error_details(output)
                
                results.append({
                    'prompt_category': test_data.get('prompt_category', 'unknown'),
                    'prompt_id': test_data.get('prompt_id', 'unknown'),
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
                    pass_at_k_scores[f'pass@{k}'] = self._pass_at_k(n, c, k)
            
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
    
    def _extract_input_line(self, test_data: Dict[str, Any]) -> str:
        """Extract input line from test case metadata"""
        try:
            metadata = test_data.get('test_case_metadata', {})
            if isinstance(metadata, str):
                metadata = literal_eval(metadata)
            return metadata.get('input', '')
        except:
            try:
                metadata = json.loads(test_data.get('test_case_metadata', '{}'))
                return metadata.get('input', '')
            except:
                return ''
    
    def _supplement_inputs_in_py(self, input_line: str, masked_code: str) -> str:
        """Supplement inputs in Python code if needed"""
        if not masked_code.startswith("inputs") and (
            'assert Solution().f(*inputs)' in masked_code or 
            'assert Solution().f(inputs)' in masked_code or 
            'assert f(*inputs)' in masked_code or 
            'assert f(inputs)' in masked_code
        ):
            return input_line + '\n' + masked_code
        return masked_code
    
    def _parse_error_details(self, error_output: str) -> Tuple[str, str]:
        """Parse error output to extract error type and code"""
        if not error_output:
            return "unknown", "UNCLASSIFIED_ERROR"
        
        error_output_lower = error_output.lower()
        
        # Compilation errors
        if "compilation failed" in error_output_lower:
            error_type = "compilation_error"
            if "cannot find symbol" in error_output_lower:
                error_code = "SYMBOL_NOT_FOUND"
            elif "';' expected" in error_output_lower:
                error_code = "SEMICOLON_EXPECTED"
            elif "incompatible types" in error_output_lower:
                error_code = "TYPE_MISMATCH"
            else:
                error_code = "OTHER_COMPILATION_ERROR"
        
        # Runtime errors
        elif "runtime error" in error_output_lower:
            error_type = "runtime_error"
            if "nullpointerexception" in error_output_lower:
                error_code = "NULL_POINTER_EXCEPTION"
            elif "arrayindexoutofboundsexception" in error_output_lower:
                error_code = "ARRAY_INDEX_OUT_OF_BOUNDS"
            elif "assertionerror" in error_output_lower:
                error_code = "ASSERTION_FAILED"
            else:
                error_code = "OTHER_RUNTIME_ERROR"
        
        # Timeout errors
        elif "timeout" in error_output_lower:
            error_type = "timeout_error"
            error_code = "EXECUTION_TIMEOUT"
        
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
    
    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        """Calculate pass@k score"""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    def _save_evaluation_results(self, results: List[Dict[str, Any]], output_file: str, 
                               generations: List[Dict[str, Any]], input_file: str):
        """Save evaluation results to output file and annotated generations file"""
        # Save evaluation results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Create hash table for quick lookup
        hash_table = {}
        for result in results:
            if result['results']:  # Only if there are actual results
                r = result['results']
                key = f"{result['problem_id']},{result['lang']},{r[0]['prompt_category']},{r[0]['prompt_id']},{r[0]['test_case_id']}"
                hash_table[key] = (result['pass_at_k'], r)
        
        # Save annotated generations file in evaluations/ directory  
        input_basename = os.path.basename(input_file)
        evaluated_file = os.path.join(os.path.dirname(output_file), input_basename)
        with open(evaluated_file, "w", encoding="utf-8") as f:
            for line in generations:
                # Add evaluation metrics to each generation
                key = f"{line['id']},{line['lang']},{line.get('prompt_category', 'unknown')},{line.get('prompt_id', 'unknown')},{line.get('test_case_idx', 1)}"
                if key in hash_table:
                    line['metrics'] = hash_table[key][0]
                    line['results'] = hash_table[key][1]
                else:
                    line['metrics'] = {}
                    line['results'] = []
                f.write(json.dumps(line) + '\n')
        
        print(f"Evaluation results saved to: {output_file}")
        print(f"Annotated generations saved to: {evaluated_file}")
    
    def _calculate_average_pass_at_1(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average pass@1 from results"""
        pass_at_1_scores = []
        for result in results:
            if 'pass@1' in result.get('pass_at_k', {}):
                pass_at_1_scores.append(result['pass_at_k']['pass@1'])
        
        if not pass_at_1_scores:
            return 0.0
        
        return sum(pass_at_1_scores) / len(pass_at_1_scores)
    
    def _calculate_existing_pass_at_1(self, output_file: str) -> float:
        """Calculate pass@1 from existing evaluation file"""
        try:
            results = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        results.append(result)
                    except:
                        continue
            
            return self._calculate_average_pass_at_1(results)
        except:
            return 0.0
    
    def print_evaluation_summary(self, results: List[Dict[str, Any]]):
        """Print evaluation summary statistics"""
        if not results:
            print("No results to summarize")
            return
        
        total_problems = len(results)
        total_success = sum(r['success_count'] for r in results)
        total_tests = sum(r['total_tests'] for r in results)
        
        java_results = [r for r in results if r['lang'].lower() == 'java']
        python_results = [r for r in results if r['lang'].lower() == 'python']
        
        print("\n" + "="*50)
        print("CODE REASONING EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Problems: {total_problems}")
        print(f"Total Test Cases: {total_tests}")
        print(f"Total Successful: {total_success}")
        
        if total_tests > 0:
            print(f"Overall Success Rate: {total_success/total_tests*100:.2f}%")
        else:
            print("Overall Success Rate: No tests executed")
        
        # Calculate overall pass@k scores
        all_pass_at_k = {}
        for k in self.k_values:
            valid_results = [r for r in results if f'pass@{k}' in r.get('pass_at_k', {})]
            if valid_results:
                avg_pass_k = np.mean([r['pass_at_k'][f'pass@{k}'] for r in valid_results])
                all_pass_at_k[f'pass@{k}'] = avg_pass_k
        
        print(f"\nOverall Pass@K Scores:")
        for k, score in all_pass_at_k.items():
            print(f"  {k}: {score:.4f}")
        
        # Language-specific statistics
        if java_results:
            java_success = sum(r['success_count'] for r in java_results)
            java_total = sum(r['total_tests'] for r in java_results)
            if java_total > 0:
                print(f"\nJava: {len(java_results)} problems, {java_success}/{java_total} tests passed ({java_success/java_total*100:.2f}%)")
        
        if python_results:
            python_success = sum(r['success_count'] for r in python_results)
            python_total = sum(r['total_tests'] for r in python_results)
            if python_total > 0:
                print(f"Python: {len(python_results)} problems, {python_success}/{python_total} tests passed ({python_success/python_total*100:.2f}%)")
        
        # Error statistics
        error_type_counts = {}
        for result in results:
            for test_result in result.get('results', []):
                if not test_result['success']:
                    error_type = test_result.get('error_type', 'unknown')
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        if error_type_counts:
            print(f"\nTop Error Types:")
            for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {error_type}: {count}")


if __name__ == "__main__":
    # Example usage
    # Input file should be in parsed/ directory
    # Output will be automatically saved in evaluations/ directory
    
    evaluator = CodeReasoningEvaluator(max_workers=8)
    
    # Evaluate from parsed/generations.jsonl 
    # Output automatically goes to evaluations/
    avg_pass_at_1 = evaluator.evaluate_file(
        input_file="parsed/output_prediction_generations.jsonl"
        # output_file will be auto-generated in evaluations/ directory
    )
    
    print(f"Final average pass@1: {avg_pass_at_1:.4f}")