from __future__ import annotations

import os
import shutil
import subprocess
import unicodedata

from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.misc.utils import ProjectCreationError
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.parsing.output_parsing import parse_output
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.eval.project_template import ProjectTemplate

from enum import Enum
THRESHOLD = 0.000002
class EvalStatus(Enum):
    CompilationError = 1
    RuntimeError = 2
    OutputError = 3
    Pass = 4

def escape_string(s: str):
    new_s = []
    for c in s:
        if c == "\\":
            new_s.append("\\\\")
        elif c == "\n":
            new_s.append("\\n")
        elif c == "\t":
            new_s.append("\\t")
        elif c == "\r":
            new_s.append("\\r")
        else:
            new_s.append(c)
    return "".join(new_s)

def normalize_unicode_string(s: str) -> str:
    """
    Normalize Unicode strings for comparison by:
    1. Unicode NFC normalization (canonical composition)
    2. Removing zero-width characters that don't affect meaning
    """
    # NFC normalization - composes characters with their combining marks
    normalized = unicodedata.normalize('NFC', s)
    
    # Remove zero-width characters that can cause comparison issues
    zero_width_chars = [
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\ufeff',  # Zero Width No-Break Space (BOM)
    ]
    
    for char in zero_width_chars:
        normalized = normalized.replace(char, '')
    
    return normalized

import re

def clean_list(input_str):
    # Use regex to extract only the numeric parts from the string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", input_str)
    # Convert the extracted numbers to floats or leave as strings depending on your requirement
    cleaned_list = [float(num) for num in numbers]
    return cleaned_list

# written mainly for java, and java-similar lang, feel free to add others
def add_imports_to_template(template_code: str, new_imports: str) -> str:
      """
      Add new import statements to the template after existing imports, 
      avoiding duplicates.
      
      Args:
          template_code: The Java template code
          new_imports: String containing new import statements (one per line)
      
      Returns:
          Template code with new unique imports added
      """
      if not new_imports or not new_imports.strip():
          return template_code

      # Extract existing imports from template
      existing_imports = set()
      template_lines = template_code.split('\n')

      for line in template_lines:
          line_stripped = line.strip()
          if line_stripped.startswith('import ') and line_stripped.endswith(';'):
              existing_imports.add(line_stripped)

      # Parse and filter new imports
      unique_new_imports = []
      for line in new_imports.strip().split('\n'):
          line_stripped = line.strip()
          if (line_stripped.startswith('import ') and
              line_stripped.endswith(';') and
              line_stripped not in existing_imports):
              unique_new_imports.append(line_stripped)

      if not unique_new_imports:
          return template_code  # No new imports to add

      # Find insertion point: after last import, before class declaration
      insert_index = -1
      for i, line in enumerate(template_lines):
          line_stripped = line.strip()
          if line_stripped.startswith('import '):
              insert_index = i + 1
          elif line_stripped.startswith('class ') and insert_index != -1:
              break

      # Insert new imports at the found position
      if insert_index != -1:
          for i, import_line in enumerate(unique_new_imports):
              template_lines.insert(insert_index + i, import_line)

      return '\n'.join(template_lines)

class Project:
    def __init__(self, path, template: ProjectTemplate):
        path = os.path.abspath(path)
        os.makedirs(path)
        self.path = path
        template_path = template.path
        self.srcs = template.srcs

        create_src_paths = [
            os.path.join(template_path, template.srcs[src_name].path)
            for src_name in template.srcs
        ]
        for root, dir, filename in os.walk(template_path, followlinks=True):
            rel_root = os.path.relpath(root, template_path)
            for name in dir:
                os.makedirs(os.path.join(self.path, rel_root, name), exist_ok=True)
            for name in filename:
                full_name = os.path.join(root, name)
                unlink = False
                for src_path in create_src_paths:
                    if os.path.samefile(full_name, src_path):
                        unlink = True
                        break
                if unlink:
                    continue
                os.symlink(full_name, os.path.join(self.path, rel_root, name))

        self.srcs = template.srcs
        self.build_cmd = template.cmds.get("build", None)
        self.run_cmd = template.cmds["run"]

    # def set_code(self, name, code):
    #     if name not in self.srcs:
    #         raise ProjectCreationError(f"Unknown code file name: {name}")
    #     src_path = os.path.join(self.path, self.srcs[name].path)
    #     with open(src_path, "w", encoding="utf-8") as f:
    #         code = self.srcs[name].code.replace("$$code$$", code)
    #         f.write(code)

    # def set_codes(self, codes: dict[str, str]):
    #     if set(codes.keys()) != set(self.srcs.keys()):
    #         raise ProjectCreationError(f"Code file names mismatch")
    #     for name, code in codes.items():
    #         self.set_code(name, code)

    def set_code(self, name, code, imports):
        if name not in self.srcs:
            raise ProjectCreationError(f"Unknown code file name: {name}")
        src_path = os.path.join(self.path, self.srcs[name].path)
        with open(src_path, "w", encoding="utf-8") as f:
            code = self.srcs[name].code.replace("$$code$$", code)
            if imports:
                code = add_imports_to_template(code, imports)
            self.test_code = code
            f.write(code)

    def set_codes(self, codes: dict[str, str], imports: str = ""):
        if set(codes.keys()) != set(self.srcs.keys()):
            raise ProjectCreationError(f"Code file names mismatch")
        for name, code in codes.items():
            self.set_code(name, code, imports)
            
    def compile(self, timeout=10):
        if self.build_cmd is None:
            return True, "Build not needed"
        try:
            ret = subprocess.run(
                self.build_cmd,
                cwd=self.path,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return True, "Build over"
        except Exception as e:
            return False, str(e)
    import subprocess
    import os


    def run(self, timeout=10):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            if len(self.run_cmd) == 1 and self.run_cmd[0].startswith("./"):
                if os.path.isfile(os.path.join(self.path, self.run_cmd[0][2:])):
                    executable = os.path.join(self.path, self.run_cmd[0][2:])
                elif os.path.isfile(os.path.join(self.path, self.run_cmd[0][2:] + ".exe")):
                    executable = os.path.join(self.path, self.run_cmd[0][2:] + ".exe")
                else:
                    raise ProjectCreationError(f"Executable {self.run_cmd[0]} not found")
                ret = subprocess.run(
                    executable,
                    cwd=self.path,
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            else:
                ret = subprocess.run(
                    self.run_cmd,
                    cwd=self.path,
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
            return True, "Run over"
        except subprocess.SubprocessError as e:
            print("Run error:", e)
            return False, str(e)

    def check_output(self):
        output_txt_path = os.path.join(self.path, "output.txt")
        if not os.path.exists(output_txt_path):
            return False, "No output.txt"
        try:
            with open(output_txt_path, "r", encoding="utf-8") as f:
                output_lines = f.read()
        except UnicodeDecodeError as e:
            return False, "Output file can't decode"
        res = parse_output(output_lines)
        if res is None:
            return False, "Output format error"
        else:
            exp_out_mismatch = False
            side_effect_error = False
            _cur_test_input, _cur_test_output, _cur_expected = None, None, None
            for test in res:
                expected_output_pairs = test[0]
                side_effect_pairs = test[1]
                for pair in expected_output_pairs:
                    expected, output = pair
                    _cur_test_output, _cur_expected = output, expected
                    if expected != output:
                        # Try Unicode normalization first
                        normalized_expected = normalize_unicode_string(expected)
                        normalized_output = normalize_unicode_string(output)
                        
                        if normalized_expected == normalized_output:
                            continue
                        
                        # Fall back to escape string comparison
                        escaped_expected = escape_string(expected)
                        escaped_output = escape_string(output)
                        
                        if escaped_expected == escaped_output:
                            continue
                        colon_of_expect = expected.rfind(":")
                        type_of_expected = expected[colon_of_expect+1:]
                        colon_of_output = output.rfind(":")
                        type_of_output = output[colon_of_output+1:]
                        if type_of_expected == 'double' and type_of_output == 'double':
                            both_double = True       
                        else:
                            both_double = False    
                        if type_of_expected == 'list<double>' and type_of_output == 'list<double>':  
                            both_double_list = True
                        else:
                            both_double_list = False         
                        if both_double:
                            expected = float(expected.split(":")[0])
                            output = float(output.split(":")[0])
                            num_decimal_places = min(str(expected)[::-1].find('.'), str(output)[::-1].find('.'))
                            # Format b to have the same number of decimal places as a
                            output = float(format(output, f".{num_decimal_places}f"))
                            THRESHOLD = 10 ** -int(num_decimal_places - 2)
                            if abs(expected - output) > THRESHOLD:
                                exp_out_mismatch = True
                                break
                        elif both_double_list:
                            expected = clean_list(expected.split(":")[0])
                            output = clean_list(output.split(":")[0])
                            num_decimal_places = min(str(expected)[::-1].find('.'), str(output)[::-1].find('.'))
                            THRESHOLD = 10 ** -int(num_decimal_places - 2)
                            break_out = False
                            for exp, out in zip(expected, output):
                                if abs(exp - out) > THRESHOLD:
                                    print("comparing", exp, out)
                                    exp_out_mismatch = True
                                    break_out = True
                            if break_out:
                                break
                        else:
                            exp_out_mismatch = True   
                            break
                for pair in side_effect_pairs:
                    before, after = pair
                    if before != after:
                        side_effect_error = True
            if exp_out_mismatch:
                return False, f"Output mismatch:\nExpected:\n{_cur_expected}\nActual:\n{_cur_test_output}"
            elif side_effect_error:
                return False, "Side-effect error"
            else:
                return True, "All Passed!"

    def read_output(self):
        output_txt_path = os.path.join(self.path, "output.txt")
        try:
            with open(output_txt_path, "r", encoding="utf-8") as f:
                output = f.read()
            return output
        except UnicodeDecodeError as e:
            return None

    def delete_folder(self):
        shutil.rmtree(self.path)

    def evaluate(self, compile_timeout=60, run_timeout=60, keep_after_eval=False, keep_when_fail=False) -> (
    EvalStatus, str):
        build_stat, msg = self.compile(timeout=compile_timeout)
        if not build_stat:
            ret_stat, msg = EvalStatus.CompilationError, msg
        else:
            run_stat, msg = self.run(timeout=run_timeout)
            if not run_stat:
                ret_stat, msg = EvalStatus.RuntimeError, msg
            else:
                output_stat, msg = self.check_output()
                if not output_stat:
                    ret_stat, msg = EvalStatus.OutputError, msg
                else:
                    ret_stat, msg = EvalStatus.Pass, "All Passed!"
        if keep_after_eval:
            pass
        elif not keep_after_eval and keep_when_fail:
            if ret_stat == EvalStatus.Pass:
                self.delete_folder()
        else:
            self.delete_folder()

        return ret_stat, msg
