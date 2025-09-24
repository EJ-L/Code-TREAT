from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from pathlib import Path
import subprocess
import tempfile
import os
import traceback
import sys
from threading import Lock
from abc import ABC, abstractmethod
import numpy as np
from extractors.tree_sitter_extraction_utils.java_tree_sitter_utils import get_unique_classes, seperate_java_files, get_java_imports
import re
# set recursion limit as 10GB
sys.setrecursionlimit(1024 * 1024 * 1024)
CodeLanguage = Literal["python", "java"]
ProblemStyle = Literal["hackerrank", "humaneval", "geeksforgeeks", "function"]


# _REL_TOL = 1e-12
# _ABS_TOL = 1e-12
_REL_TOL = 1e-6
_ABS_TOL = 1e-9
def clean_output(s: str) -> str:
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip() not in {"~"})

def line_to_float_array(line: str) -> Optional[np.ndarray]:
    try:
        # Extract all numbers (int/float) regardless of separators
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', line.strip())
        if not numbers:
            return None
        return np.array([float(x) for x in numbers])
    except (ValueError, OverflowError):
        return None

# def outputs_equal(pred: str, expect: str) -> bool:
#     pred, expect = clean_output(pred), clean_output(expect)
#     p_lines = [ln.strip() if isinstance(ln, str) else ln for ln in str(pred).strip().splitlines()]
#     e_lines = [ln.strip()  if isinstance(ln, str) else ln for ln in str(expect).strip().splitlines()]
#     if len(p_lines) != len(e_lines):
#         return False

#     for pl, el in zip(p_lines, e_lines):
#         if pl == el:
#             continue
#         p_arr = line_to_float_array(pl)
#         e_arr = line_to_float_array(el)
#         if p_arr is None or e_arr is None:
#             if pl != el:
#                 return False
#             continue
#         if p_arr.shape != e_arr.shape:
#             return False
#         if not np.allclose(p_arr, e_arr, rtol=_REL_TOL, atol=_ABS_TOL):
#             return False
#     return True

def _decimals_in_line(s: str) -> int:
    # Use the *maximum* decimals among floats on the line
    decs = 0
    for m in re.finditer(r'-?\d+(?:\.(\d+))?(?:[eE][+-]?\d+)?', s):
        if m.group(1):
            decs = max(decs, len(m.group(1)))
    return decs

def outputs_equal(pred: str, expect: str) -> bool:
    p_lines = [str(ln).strip() for ln in str(pred).strip().splitlines()]
    e_lines = [str(ln).strip() for ln in str(expect).strip().splitlines()]
    if len(p_lines) != len(e_lines):
        return False

    for pl, el in zip(p_lines, e_lines):
        if pl == el:
            continue

        p_arr = line_to_float_array(pl)
        e_arr = line_to_float_array(el)
        if p_arr is None or e_arr is None:
            if pl != el:
                return False
            continue
        if p_arr.shape != e_arr.shape:
            return False

        # half-ulp at the last printed decimal (min 6dp to match HackerRank style)
        dp = max(_decimals_in_line(el), 6)
        atol = 0.5 * (10 ** (-dp))
        if not np.allclose(p_arr, e_arr, rtol=1e-12, atol=atol):
            return False
    return True
def outputs_equal(pred: str, expect: str, dp: int = 6) -> bool:
    pred, expect = clean_output(pred), clean_output(expect)

    p_lines = [ln.strip() for ln in str(pred).strip().splitlines()]
    e_lines = [ln.strip() for ln in str(expect).strip().splitlines()]
    if len(p_lines) != len(e_lines):
        return False

    for pl, el in zip(p_lines, e_lines):
        p_arr = line_to_float_array(pl)
        e_arr = line_to_float_array(el)
        if p_arr is None or e_arr is None:
            # fallback to string equality
            if pl != el:
                return False
            continue
        if p_arr.shape != e_arr.shape:
            return False

        # round both arrays to dp decimals
        if not np.allclose(np.round(p_arr, dp), np.round(e_arr, dp)):
            return False
    return True

class CodeExecutor:
    def __init__(self, python_executable: str = None) -> None:
        self.executor = {
            'python': PythonExecutor(python_executable),
            'java': JavaExecutor(),
        }
        
    def run_code(self, lang: str = 'python', code: str = "", test_inputs: List[str] = [], test_outputs: List[str] = [], timeout: int = 10) -> Dict[str, Any]:
        if lang not in self.executor:
            raise ValueError(f"Unsupported language: {lang}")
        return self.executor[lang.lower()].run_code(code, test_inputs, test_outputs, timeout)

    def get_expected(self, lang: str, code: str, test_inputs: List[str], timeout: int = 10) -> Union[Tuple[bool, List[str]], Tuple[None, str], List[str]]:
        if lang not in self.executor:
            raise ValueError(f"Unsupported language: {lang}")
        return self.executor[lang.lower()].get_expected(code, test_inputs, timeout)

class LanguageExecutor(ABC):
    """Abstract base class for language-specific code execution"""
    
    @abstractmethod
    def run_code(self, code: str, test_inputs: List[str] = [], test_outputs: List[str] = [], timeout: int = 10) -> Dict[str, Any]: 
        pass
    
    @abstractmethod
    def get_expected(self, code: str, test_inputs: List[str], timeout: int = 10) -> Union[Tuple[bool, List[str]], Tuple[None, str], List[str]]:
        pass

class PythonExecutor(LanguageExecutor):
    def __init__(self, python_executable: str = None):
        self.python_executable = python_executable or sys.executable
    
    def get_expected(self, code: str, test_inputs: List[str], timeout: int = 10) -> Tuple[Union[bool, None], Union[List[str], str]]:
        code_path = None
        output_path = None
        process = None
        original_dir = os.getcwd()
        try:
            # Create temporary directory and save code
            with tempfile.TemporaryDirectory() as temp_dir:
                code_path = os.path.join(temp_dir, "script.py")
                with open(code_path, "w") as f:
                    f.write(code)
                output_path = os.path.join(temp_dir, "output.txt")
                
                # Set up environment
                env = os.environ.copy()
                env["OUTPUT_PATH"] = output_path
                env["PYTHONPATH"] = os.getcwd()
                
                results = []
                for test_input in test_inputs:
                    try:
                        process = subprocess.Popen(
                            [self.python_executable, code_path],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            env=env,
                            cwd=temp_dir,
                        )
                        try:
                            stdout, stderr = process.communicate(input=test_input, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            return None, "Timeout"
                        except Exception as e:
                            return None, str(e)

                        # move to the input, allow some invalid inputs, as not inputs are invalid
                        if process.returncode != 0:
                            continue

                        actual_output = stdout
                        if os.path.exists(output_path):
                            with open(output_path, 'r') as f:
                                file_output = f.read()
                                if file_output:
                                    actual_output = file_output
                    
                        results.append(actual_output)

                    except Exception as e:
                        return None, str(e)
                        
                return True, results
        finally:
            if process and process.poll() is None:
                try:
                    process.kill()
                except:
                    pass
            os.chdir(original_dir)

    
    def run_code(self, code: str, test_inputs: List[str], expected_outputs: List[str], timeout: int = 10) -> Dict[str, Any]:
        code_path = None
        output_path = None
        process = None
        original_dir = os.getcwd()
        try:
            # Create temporary directory and save code
            with tempfile.TemporaryDirectory() as temp_dir:
                code_path = os.path.join(temp_dir, "script.py")
                with open(code_path, "w") as f:
                    f.write(code)
                output_path = os.path.join(temp_dir, "output.txt")
                # Set up environment
                env = os.environ.copy()
                env["OUTPUT_PATH"] = output_path
                env["PYTHONPATH"] = os.getcwd()
                
                results = []
                for test_input, expected_output in zip(test_inputs, expected_outputs):
                    try:
                        process = subprocess.Popen(
                            [self.python_executable, code_path],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            env=env,
                            cwd=temp_dir,
                        )
                        try:
                            stdout, stderr = process.communicate(input=test_input, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            if process:
                                process.kill()
                            return {
                                "passed": False,
                                "msg": "Code execution timed out (possible infinite loop)"
                            }
                        except Exception as e:
                            if process:
                                process.kill()

                            return {
                                "passed": False,
                                "msg": f"Code execution error: {str(e)}"
                            }
                        
                        if process.returncode != 0:
                            return {
                                "passed": False,
                                "inputs": test_input,
                                "msg": f"Runtime error: {stderr}"
                            }

                        actual_output = stdout
                        if os.path.exists(output_path):
                            with open(output_path, 'r') as f:
                                file_output = f.read()
                                if file_output:
                                    actual_output = file_output
                    
                        # not evaluation mode
                        if expected_output is None:
                            passed = False
                        else:
                            passed = outputs_equal(actual_output, expected_output)
                        if not passed:
                            return {
                                "passed": False,
                                "inputs": test_input,
                                "actual_output": actual_output,
                                "expected_output": expected_output,
                                "msg": "wrong answer",
                            }

                    except Exception as e:
                        return {
                            "passed": False,
                            "inputs": test_input,
                            "error": f"{type(e).__name__}: {str(e)}",
                            "traceback": traceback.format_exc()
                        }
                        
                return {
                    "passed": True,
                    "msg": "all passed"
                }
        finally:
            if process and process.poll() is None:
                try:
                    process.kill()
                except:
                    pass
            os.chdir(original_dir)

class JavaExecutor(LanguageExecutor):
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "java_execution"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def get_expected(self, code: str, test_inputs: List[str], timeout: int = 10, _script_name: str = 'Main') -> Union[Tuple[None, str], List[str]]:
        try:
            with tempfile.TemporaryDirectory() as exec_dir:
                exec_path = Path(exec_dir)
                imports = get_java_imports(code)
                if imports:
                    imports = imports[0]
                # seperate java files according to modifiers
                class_dict = get_unique_classes(code)
                java_files_dict = seperate_java_files(class_dict)
                if len(java_files_dict) != 0:
                    return {
                        'passed': False,
                        'error': f"Only 1 public class and 1 temp file should be find, dict:\n{java_files_dict}code:{code}"
                    }
                compile_cmd = ['javac']
                script_name = None
                for file_name, file_content in java_files_dict.items():
                    java_file = exec_path / file_name
                    java_file.write_text(imports + '\n' + file_content)
                    compile_cmd += [file_name]
                    script_name = file_name.split('.')[0] if 'Temp' not in file_name else script_name

                output_file = exec_path / "output.txt"
                
                # Compile once for all test cases
                try:
                    compile_result = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        cwd=exec_path,
                        timeout=10
                    )
                except subprocess.TimeoutExpired:
                    return None, "Timeout"
                
                if compile_result.returncode != 0:
                    return None, str(compile_result.stderr)
                
                # Set up environment
                env = os.environ.copy()
                env["OUTPUT_PATH"] = str(output_file)
                
                results = []
                for test_input in test_inputs:
                    try:
                        run_result = subprocess.run(
                            ['java', '-cp', str(exec_path), script_name],
                            input=test_input,
                            capture_output=True,
                            text=True,
                            cwd=exec_path,
                            env=env,
                            timeout=5
                        )
                    except subprocess.TimeoutExpired:
                        return None, "Timeout"

                    if run_result.returncode != 0:
                        error_msg = run_result.stderr
                        return None, error_msg
                    
                    actual_output = run_result.stdout.strip()
                    if output_file.exists():
                        file_output = output_file.read_text().strip()
                        if file_output:
                            actual_output = file_output
                    
                    results.append(actual_output)

                return results
            
        except Exception as e:
            return None, e

    def run_code(self, code: str, test_inputs: List[str], expected_outputs: List[str], timeout: int = 10) -> Dict[str, Any]:
        """Run Java code with timeout"""
        try:
            with tempfile.TemporaryDirectory() as exec_dir:
                exec_path = Path(exec_dir)
                imports = get_java_imports(code)
                if imports:
                    imports = imports[0]
                # seperate java files according to modifiers
                class_dict = get_unique_classes(code)
                java_files_dict = seperate_java_files(class_dict)
                if not (len(java_files_dict) <= 2 and len(java_files_dict) != 0):
                    return {
                        'passed': False,
                        'error': f"Only 1 public class and 1 temp file should be find, dict:\n{java_files_dict}code:{code}"
                    }
                compile_cmd = ['javac']
                script_name = None
                for file_name, file_content in java_files_dict.items():
                    java_file = exec_path / file_name
                    java_file.write_text(imports + '\n' + file_content)
                    compile_cmd += [file_name]
                    script_name = file_name.split('.')[0] if 'Temp' not in file_name else script_name
                output_file = exec_path / "output.txt"
                
                # Compile once for all test cases
                try:
                    compile_result = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        cwd=exec_path,
                        timeout=10
                    )
                except subprocess.TimeoutExpired:
                    return {
                        'passed': False,
                        'error': 'Compilation timed out after 10 seconds'
                    }
                
                if compile_result.returncode != 0:
                    return {
                        'passed': False,
                        'error': f'Compilation error: {compile_result.stderr}'
                    }
                
                # Set up environment
                env = os.environ.copy()
                env["OUTPUT_PATH"] = str(output_file)
                
                for test_input, expected_output in zip(test_inputs, expected_outputs):
                    try:
                        run_result = subprocess.run(
                            ['java', '-cp', str(exec_path), script_name],
                            input=test_input,
                            capture_output=True,
                            text=True,
                            cwd=exec_path,
                            env=env,
                            timeout=5
                        )
                    except subprocess.TimeoutExpired:
                        return {
                            'passed': False,
                            'msg': f'Execution timed out after {timeout} seconds'
                        }

                    if run_result.returncode != 0:
                        error_msg = run_result.stderr
                        
                        return {
                            'passed': False,
                            'msg': f'Runtime error: {error_msg}'
                        }
                    
                    actual_output = run_result.stdout.strip()
                    if output_file.exists():
                        file_output = output_file.read_text().strip()
                        if file_output:
                            actual_output = file_output
                            
                    # Early return if output doesn't match
                    if not outputs_equal(actual_output, expected_output):
                        return {
                            'passed': False,
                            'input': test_input,
                            'actual_output': actual_output,
                            'expected_output': expected_output.strip()
                        }

                return {
                    "passed": True,
                    "msg": "all passed!"
                }
                
        except Exception as e:
            return {
                'passed': False,
                'msg': f'Execution error: {str(e)}'
            }

