"""
Code Reasoning Extractor with 2-step parsing approach:
1. Regex/pattern parsing first (reusing old framework logic)
2. LLM parsing for failed cases with progress tracking and resume capability
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
from ast import literal_eval
import re

# Import model interface (assuming similar structure to old framework)
try:
    from models.model_list import MODELS
    DEFAULT_PARSING_MODEL = MODELS.get('gpt-4o-2024-11-20')
except ImportError:
    DEFAULT_PARSING_MODEL = None
    print("Warning: Could not import MODELS. Please set parsing_model explicitly.")


class CodeReasoningExtractor:
    """
    2-step code reasoning response extractor:
    Step 1: Regex/pattern parsing (from old framework)
    Step 2: LLM parsing for failed cases with progress tracking
    """
    
    def __init__(self, parsing_model=None, max_responses_to_parse=None, max_workers=4, max_retries=3):
        """
        Initialize the extractor
        
        Args:
            parsing_model: Model to use for LLM parsing (defaults to gpt-4o-2024-11-20)
            max_responses_to_parse: Max number of responses to parse from each response list (defaults to all)
            max_workers: Number of parallel workers for LLM parsing
            max_retries: Maximum retries for LLM parsing failures
        """
        self.parsing_model = parsing_model or DEFAULT_PARSING_MODEL
        self.max_responses_to_parse = max_responses_to_parse
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._progress_lock = threading.Lock()
        
        if self.parsing_model is None:
            raise ValueError("No parsing model provided. Please set parsing_model parameter.")
    
    def extract_from_file(self, input_file: str, output_file: str = None) -> str:
        """
        Extract parsed responses from input file using 2-step approach
        
        Args:
            input_file: Path to input JSONL file with raw responses (in predictions/<model_name>/)
            output_file: Path to output JSONL file (optional, auto-generated in parsed/)
            
        Returns:
            Path to the output file
        """
        # Get the root directory (where predictions/ folder is located)
        # Assuming input_file is in predictions/<model_name>/file.jsonl
        predictions_dir = os.path.dirname(input_file)
        if os.path.basename(predictions_dir).startswith('predictions'):
            # input_file is directly in predictions/ folder
            root_dir = os.path.dirname(predictions_dir)
        else:
            # input_file is in predictions/<model_name>/ folder
            root_dir = os.path.dirname(os.path.dirname(input_file))
        
        # Setup directories following framework structure
        parsed_dir = os.path.join(root_dir, "parsed")
        os.makedirs(parsed_dir, exist_ok=True)
        
        # Generate output file path if not provided
        if output_file is None:
            input_basename = os.path.basename(input_file)
            output_file = os.path.join(parsed_dir, input_basename)
        
        # Progress file path in parsed/ directory
        input_basename = os.path.basename(input_file).replace('.jsonl', '')
        progress_file = os.path.join(parsed_dir, f"{input_basename}_progress.json")
        
        # Load existing progress
        progress_data = self._load_progress(progress_file)
        
        # Process file
        print(f"Processing {input_file} -> {output_file}")
        print(f"Progress tracking: {progress_file}")
        
        processed_lines = []
        total_lines = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            total_lines = len(lines)
        
        for line_num, line in enumerate(lines, 1):
            try:
                json_line = json.loads(line.strip())
                
                # Generate ref_key (matching your framework's ref_key generation)
                ref_key = self._generate_ref_key(json_line)
                
                # Check if already processed
                if progress_data.get(ref_key, {}).get('processed', False):
                    print(f"Skipping already processed ref_key: {ref_key}")
                    processed_lines.append(json_line)
                    continue
                
                print(f"Processing line {line_num}/{total_lines}: {ref_key}")
                
                # Extract and process responses
                processed_line = self._process_single_line(json_line, ref_key, progress_data)
                processed_lines.append(processed_line)
                
                # Save progress
                self._save_progress(progress_file, progress_data)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                processed_lines.append(json_line)  # Keep original line on error
                continue
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in processed_lines:
                f_out.write(json.dumps(line) + '\n')
        
        print(f"Completed processing. Output written to: {output_file}")
        return output_file
    
    def _generate_ref_key(self, json_line: Dict[str, Any]) -> str:
        """Generate ref_key similar to RecordManager.make_ref_key"""
        # Adapt this based on your actual ref_key generation logic
        dataset = json_line.get('dataset', 'unknown')
        model_name = json_line.get('model_name', 'unknown')
        problem_id = json_line.get('id', 'unknown')
        test_case_idx = json_line.get('test_case_idx', 1)
        lang = json_line.get('lang', 'unknown')
        prompt_category = json_line.get('prompt_category', 'unknown')
        prompt_id = json_line.get('prompt_id', 'unknown')
        
        return f"{dataset}_{model_name}_{problem_id}_{test_case_idx}_{lang}_{prompt_category}_{prompt_id}"
    
    def _process_single_line(self, json_line: Dict[str, Any], ref_key: str, progress_data: Dict) -> Dict[str, Any]:
        """Process a single line with 2-step parsing approach"""
        
        # Get responses to parse
        all_responses = json_line.get('response', [])
        if not all_responses:
            return json_line
        
        # Limit to first-N responses if specified
        responses_to_parse = all_responses
        if self.max_responses_to_parse is not None:
            responses_to_parse = all_responses[:self.max_responses_to_parse]
        
        # Get task details
        task = json_line.get('task', 'output_prediction')
        lang = json_line.get('lang', 'python')
        
        # Get masked query for parsing
        try:
            test_case_metadata = json_line.get('test_case_metadata', {})
            if isinstance(test_case_metadata, str):
                test_case_metadata = literal_eval(test_case_metadata)
            masked_query = test_case_metadata.get(task, '')
        except:
            masked_query = ''
        
        # STEP 1: Regex/Pattern Parsing
        print(f"Step 1: Regex parsing {len(responses_to_parse)} responses")
        parsed_responses, parsed_cases = self._regex_parse_responses(
            responses_to_parse, masked_query, task.split("_")[0], lang
        )
        
        # STEP 2: LLM Parsing for Failed Cases
        print(f"Step 2: Checking for LLM parsing needed")
        if self._has_map_issues(parsed_responses, parsed_cases, lang):
            print(f"Found issues, applying LLM parsing")
            llm_parsed_responses, llm_parsed_cases = self._llm_parse_failed_cases(
                json_line.get('code/function', ''), responses_to_parse, 
                parsed_responses, parsed_cases, task, lang
            )
            final_responses = llm_parsed_responses
            final_cases = llm_parsed_cases
        else:
            final_responses = parsed_responses
            final_cases = parsed_cases
        
        # Update progress
        with self._progress_lock:
            progress_data[ref_key] = {
                'total_responses': len(all_responses),
                'max_to_parse': len(responses_to_parse),
                'processed': True,
                'regex_parsed_responses': parsed_responses,
                'regex_parsed_cases': parsed_cases,
                'final_parsed_responses': final_responses,
                'final_parsed_cases': final_cases
            }
        
        # Update json_line with parsed results
        result_line = json_line.copy()
        result_line['parsed_response'] = final_responses
        result_line['parsed_case'] = final_cases
        
        return result_line
    
    def _regex_parse_responses(self, resp: List[str], masked_query: str, mode: str = "input", lang: str = "python") -> Tuple[List[str], List[int]]:
        """
        STEP 1: Regex/pattern parsing - exact copy from old framework's parse_response function
        """
        parsed_responses = []
        parsed_case = []
        
        if mode == "input":
            key = "input_prediction"
        else:
            key = "output_prediction"
            
        if lang == 'java':
            # Java parsing logic from old framework
            for r in resp:
                r = self._remove_thinking_tag(r)
                if "```" in r and ("```java" in r or "```python" in r):
                    success = False
                    _tuple = self._parse_java_md(r)
                    if _tuple is not None:
                        content, success = _tuple
                    if success:
                        parsed_responses.append(content)
                        parsed_case.append(2 if content.lstrip().startswith("public static void main") else 5)
                        continue
                try:
                    json_r = self._grab_prediction(r) or self._parse_json_md(r)
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
                    parsed_responses.append(r)
                    parsed_case.append(-1)
                    
        elif lang == 'python':
            # Python parsing logic from old framework
            for r in resp:        
                r = self._remove_thinking_tag(r)
                if r.strip() == "":      
                    parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                    parsed_case.append(-2)
                    continue
                from_json = False
                successful_response = None
                parsed_response = self._extract_prediction(r, True)
                
                if parsed_response:
                    successful_response = parsed_response[0][key]
                    from_json = True
                else:
                    _tuple = self._parse_python_md(r)
                    if _tuple is not None:
                        content, success = _tuple
                        if success:
                            successful_response = content[0]
                            
                if mode == 'input':
                    if successful_response:
                        final_parse, idx = self._handle_input_prediction_response(successful_response, from_json)
                        parsed_responses.append(masked_query.replace("??", final_parse, 1))
                        parsed_case.append(idx)
                    else:
                        parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                        parsed_case.append(-2)
                        
                if mode == 'output':
                    if successful_response:
                        final_parse, idx = self._handle_output_prediction_response(successful_response, from_json)
                        parsed_responses.append(masked_query.replace("??", final_parse, 1))
                        parsed_case.append(idx)
                    else:
                        parsed_responses.append(masked_query.replace("??", "<ERROR>", 1))
                        parsed_case.append(-2)
        else:
            raise ValueError('Language apart from python and java are not supported')

        return parsed_responses, parsed_case
    
    def _has_map_issues(self, parsed_responses: List[str], parsed_cases: List[int], lang: str) -> bool:
        """
        Check if LLM parsing is needed - from old framework's has_map_issues logic
        """
        if lang == 'java':
            # Java: cases -1, 3, 4, 5 indicate regex parsing failures
            return any(case in [-1, 3, 4, 5] for case in parsed_cases)
        elif lang == 'python':
            # Python: cases -2, -1 indicate regex parsing failures  
            return any(case in [-2, -1] for case in parsed_cases)
        return False
    
    def _llm_parse_failed_cases(self, masked_code: str, original_responses: List[str], 
                               parsed_responses: List[str], parsed_cases: List[int],
                               task: str, lang: str) -> Tuple[List[str], List[int]]:
        """
        STEP 2: LLM parsing for failed cases with parallel processing
        """
        failed_indices = []
        for i, case in enumerate(parsed_cases):
            if lang == 'java' and case in [-1, 3, 4, 5]:
                failed_indices.append(i)
            elif lang == 'python' and case in [-2, -1]:
                failed_indices.append(i)
        
        if not failed_indices:
            return parsed_responses, parsed_cases
        
        print(f"LLM parsing needed for {len(failed_indices)} failed cases")
        
        # Prepare data for parallel processing
        llm_args = []
        for idx in failed_indices:
            llm_args.append((
                masked_code,
                original_responses[idx],
                parsed_cases[idx],
                task,
                lang
            ))
        
        # Process with ThreadPoolExecutor
        llm_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_single_llm_parse, llm_args))
            llm_results = results
        
        # Update parsed responses with LLM results
        final_responses = parsed_responses.copy()
        final_cases = parsed_cases.copy()
        
        for i, (result_response, result_case) in enumerate(llm_results):
            if result_response is not None:
                actual_idx = failed_indices[i]
                final_responses[actual_idx] = result_response
                final_cases[actual_idx] = result_case
        
        return final_responses, final_cases
    
    def _process_single_llm_parse(self, args: Tuple) -> Tuple[Optional[str], int]:
        """Process a single LLM parsing request with retries"""
        masked_code, llm_response, case, task, lang = args
        
        for attempt in range(self.max_retries):
            try:
                if lang == 'java':
                    return self._llm_parse_java(masked_code, llm_response, case, task)
                elif lang == 'python':
                    return self._llm_parse_python(masked_code, llm_response, case, task)
            except Exception as e:
                print(f"LLM parsing attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return error markers on final failure
                    return "", -2
        
        return "", -2
    
    def _llm_parse_java(self, masked_code: str, llm_response: str, case: int, task: str) -> Tuple[str, int]:
        """LLM parsing for Java - adapted from old framework"""
        # Use the same prompt logic as old framework's prompt_selector
        messages = self._build_java_llm_prompt(masked_code, llm_response, case)
        
        try:
            response = self.parsing_model.chat(messages, n=1)
            result = self._handle_java_llm_response(response[0])
            
            # Handle "NO" as a valid empty string result
            if result is None or result.lower() == "no":
                return "", 0  # Success with empty string
            
            if result.find("??") != -1:
                return "", -1  # Still has placeholders, this is a failure
            
            if not result.startswith('public static void main'):
                return "", -1  # Not a valid main method, this is a failure
                
            return result, 0
            
        except Exception as e:
            print(f"Java LLM parsing error: {e}")
            return "", -2
    
    def _llm_parse_python(self, masked_code: str, llm_response: str, case: int, task: str) -> Tuple[str, int]:
        """LLM parsing for Python - adapted from old framework"""
        # Use the same prompt as old framework's Python LLM parsing
        prompt = self._build_python_llm_prompt(masked_code, llm_response)
        
        try:
            response = self.parsing_model.chat(prompt, n=1)
            result = self._handle_python_llm_response(response[0])
            
            # Handle "NO" as a valid empty string result
            if result == "" or result.lower() == "no":
                return "", 0  # Success with empty string
            
            if result.find("assert") == -1:
                return "", -1  # No assert statement, this is a failure
                
            return result, 0
            
        except Exception as e:
            print(f"Python LLM parsing error: {e}")
            return "", -2
    
    # Helper methods from old framework (regex parsing utilities)
    
    def _remove_thinking_tag(self, response: str) -> str:
        """Remove thinking tags from response"""
        try:
            response = response.split("</think>")[-1]
        except:
            pass
        return response
    
    def _grab_prediction(self, text: str) -> Optional[Tuple[str, str]]:
        """Grab prediction from JSON-like text"""
        _HEAD_RE = re.compile(r'\{\s*"(input|output)[ _]prediction"\s*:\s*', re.I)
        m = _HEAD_RE.search(text)
        if not m:
            return None

        start = m.start()
        i = m.end()
        depth = 1
        in_str, esc = False, False

        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        fragment = text[start : i + 1]
                        value_part = text[m.end() : i]
                        return fragment, value_part.strip()
            i += 1
        return None
    
    def _parse_json_md(self, markdown_text: str):
        """Parse JSON from markdown"""
        pattern = r"```json\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        if matches and len(matches) == 1:
            return matches[0], True
        else:
            return None
    
    def _parse_java_md(self, markdown_text: str):
        """Parse Java code from markdown"""
        pattern = r"```java\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        if matches and len(matches) == 1:
            return matches[0], True
        elif matches:
            for m in matches:
                if m.strip().startswith("public static void main"):
                    return m, True
        else:
            return None, False
    
    def _parse_python_md(self, markdown_text: str):
        """Parse Python code from markdown"""
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        if matches and len(matches) == 1:
            return [matches[0]], True
        else:
            return None, False
    
    def _extract_prediction(self, response: str, return_all: bool = True):
        """Extract prediction from response - placeholder for old framework logic"""
        # This should implement the same logic as old framework's extract_prediction
        # For now, return None to indicate no extraction
        return None
    
    def _handle_input_prediction_response(self, response: str, from_json: bool) -> Tuple[str, int]:
        """Handle input prediction response - placeholder for old framework logic"""
        # This should implement the same logic as old framework's handle_input_prediction_response
        return str(response), 1 if from_json else 2
    
    def _handle_output_prediction_response(self, response: str, from_json: bool) -> Tuple[str, int]:
        """Handle output prediction response - placeholder for old framework logic"""
        # This should implement the same logic as old framework's handle_output_prediction_response
        return str(response), 1 if from_json else 2
    
    def _build_java_llm_prompt(self, masked_code: str, llm_response: str, case: int) -> List[Dict[str, str]]:
        """Build Java LLM prompt - adapted from old framework"""
        system_msg = {
            "role": "system",
            "content": (
                "You are an assistant that extracts the `public static void main` method from LLM responses. "
                "Your task is to find and return ONLY the `public static void main` method from the LLM output. "
                "\n\nInstructions:"
                "\n- If the LLM output contains a complete Java class, extract ONLY the `public static void main` method"
                "\n- If the LLM output is JSON like {\"output_prediction\": \"value\"}, extract the value and map it to any ?? in the provided template"
                "\n- If the LLM output is plain text/code, look for the main method or use it to replace ?? in the template"
                "\n- Return ONLY the `public static void main` method, not the entire class"
                "\n- Do NOT modify the extracted content - use it exactly as found"
                "\n- If you cannot find a main method or extract meaningful content, output \"NO\""
                "\n- If you somehow find only python written code instead of java code, do not translate and simply output \"NO\""
            )
        }
        
        instructions = {
            5: "The LLM output contains a complete Java class with a main method. Extract ONLY the `public static void main` method from the full class. Return just the main method, not the entire class.",
            4: "The LLM output is JSON-like (e.g., {\"output_prediction\": \"[3, 3]\"}). Extract the value from the JSON field and replace any ?? in the provided Java template with it. Return the complete main method.",
            3: "The LLM output may contain a complete Java class or other formats. If it contains a main method, extract ONLY that method. If it's other content, map it to replace ?? in the provided template.",
            -1: "Extract the main method from the LLM output if present, or map the content to replace ?? in the template. Return only the main method."
        }
        
        instruction = instructions.get(case, instructions[-1])
        
        user_msg = {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"### Code Snippet:\n{masked_code}\n\n"
                f"### LLM Output:\n{llm_response}"
            )
        }
        
        return [system_msg, user_msg]
    
    def _build_python_llm_prompt(self, masked_code: str, llm_response: str) -> List[Dict[str, str]]:
        """Build Python LLM prompt - adapted from old framework"""
        prompt_template = """
We are required to do the evaluation of the LLM's ability in code generation.

You will receive two inputs:

1. Original code: a Python snippet whose assertion line uses `??` to mark missing inputs or expected output.
2. LLM prediction: the model's response, which may include reasoning and/or a completed assertion line, or even a incomplete/complete json object containing the model's prediction.

Your task either:
1. Extract exactly the completed assertion line from the LLM prediction.
2. Match the filled-in values back to the original code using the correct variable names.

You must:
- **not** modify any other part of the original code or fix any other errors.
- **not** output any additional text or formatting. If no valid assertion line is found, return an empty string: `""`.

When reading the LLM's prediction:
- There might be incomplete json objects, you should try your best to extract the prediction, if impossible, simple return `""`.
- If it gives the direct answer, or giving the whole assertion line, you can just match them using the above rules.
- If it gives the reasoning, you should still try to match the inputs, or expected_outputs, but **not** to help the LLM complete, or correct their wrong/incomplete predictions.
- If in the reasoning, when no final conclusion regarding `inputs` or `expected_output` value are made, simple return an empty string: `""`.
- For input prediction, either extract the input parameter values (inside function call), or map the value to the appropriate `inputs = ??`
- For output prediction, either extract the expected output value (after ==),or the value to the appropriate `expected_output = ??`
- Lastly, please remember not to give your prediction.

Output specification:
Return only the completed assertion line.

Here is the original code (with incomplete assertion line):
{code}

Here is the LLM's prediction:
{prediction}
"""
        
        user_msg = {
            "role": "user",
            "content": prompt_template.format(code=masked_code, prediction=llm_response)
        }
        
        return [user_msg]
    
    def _handle_java_llm_response(self, response: str) -> str:
        """Handle Java LLM response"""
        if "```java" in response:
            return self._parse_java_md(response)[0] if self._parse_java_md(response) else response
        return response
    
    def _handle_python_llm_response(self, response: str) -> str:
        """Handle Python LLM response"""
        if "```" in response:
            response = response.replace("```python\n", "")
            response = response.replace("```", "")
        return response
    
    def _load_progress(self, progress_file: str) -> Dict:
        """Load progress from JSON file"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading progress file {progress_file}: {e}")
                return {}
        return {}
    
    def _save_progress(self, progress_file: str, progress_data: Dict):
        """Save progress to JSON file"""
        try:
            with self._progress_lock:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Error saving progress file {progress_file}: {e}")


if __name__ == "__main__":
    # Example usage
    # Input file should be in predictions/<model_name>/ directory
    # Output will be automatically saved in parsed/ directory
    
    extractor = CodeReasoningExtractor(max_responses_to_parse=5)
    
    # Extract from predictions/<model_name>/generations.jsonl
    # Output automatically goes to parsed/generations.jsonl
    extractor.extract_from_file(
        input_file="predictions/gpt-4o/output_prediction_generations.jsonl"
        # output_file will be auto-generated in parsed/ directory
    )