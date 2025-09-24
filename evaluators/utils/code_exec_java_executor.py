#!/usr/bin/env python3
import os
import sys
import json
import re
import shutil
import subprocess
import tempfile
import threading
import uuid


from tree_sitter_language_pack import get_language, get_parser
JAVA_LANG = get_language('java')
JAVA_PARSER = get_parser('java')
TREE_SITTER_AVAILABLE = True
    

class JavaExecutor:
    """Enhanced Java executor using tree-sitter for precise code injection"""
    
    def __init__(self, masked_test_code, cls_name="Solution"):
        self.masked_test_code = masked_test_code
        self.cls_name = cls_name
        self.temp_dir = None
        
    def inject_main_code(self, main_code):
        """Inject main method into the masked test code using tree-sitter or regex fallback"""
        if TREE_SITTER_AVAILABLE:
            return self._inject_main_code_tree_sitter(main_code)
        else:
            return self._inject_main_code_regex(main_code)
    
    def _inject_main_code_tree_sitter(self, main_code):
        """Use tree-sitter to inject main method precisely"""
        try:
            byte_src, cls_pos, last_method_end_pos = self.find_cls_body()
            new_byte_src = byte_src[:last_method_end_pos] + bytes(main_code, 'utf-8') + byte_src[last_method_end_pos:]
            return new_byte_src.decode('utf-8')
            # # make the test code mutable
            # src_masked_test_code = bytearray(self.masked_test_code, 'utf-8')

            # # replace the class body region with your updated source
            # start, end = cls_pos
            # src_masked_test_code[start:end] = new_byte_src
            # return src_masked_test_code.decode('utf-8')
        except Exception as e:
            print(f"Tree-sitter injection failed: {e}, falling back to regex")
            return self._inject_main_code_regex(main_code)
        
    def find_cls_body(self):
        """Find class body end position using tree-sitter"""
        src = bytes(self.masked_test_code, 'utf-8')
        query = f"""
        (class_declaration
            name: (identifier) @cls_name (#eq? @cls_name "{self.cls_name}")
            body: (
                class_body
                (method_declaration) @method
            )
        ) @full_cls
        """
        root = JAVA_PARSER.parse(src).root_node
        extraction_query = JAVA_LANG.query(query)
        matches = extraction_query.matches(root)
        method_positions = []
        
        for match in matches:
            pattern_idx, match_dict = match
            method_positions.append(match_dict['method'][0].end_byte)
        cls_pos = (matches[0][1]['full_cls'][0].start_byte, matches[0][1]['full_cls'][0].end_byte)

        if not method_positions:
            # Fallback: find class end
            class_query = f"""
            (class_declaration
                name: (identifier) @cls_name (#eq? @cls_name "{self.cls_name}")
                body: (class_body) @body
            )
            """
            class_matches = JAVA_LANG.query(class_query).matches(root)
            if class_matches:
                for match in class_matches:
                    pattern_idx, match_dict = match
                    # Insert before closing brace
                    class_body = match_dict['body'][0]
                    return src, class_body.end_byte - 1  # Before closing brace
            
            # Last resort: find end of last method in any class
            method_query = "(method_declaration) @method"
            method_matches = JAVA_LANG.query(method_query).matches(root)
            if method_matches:
                method_positions = [match_dict['method'][0].end_byte for _, match_dict in method_matches]
        
        method_positions.sort()
        return src, cls_pos, method_positions[-1] if method_positions else len(src) - 1
    
    def _inject_main_code_regex(self, main_code):
        """Fallback regex-based injection"""
        # Find the last closing brace (should be class closing brace)
        last_brace_idx = self.masked_test_code.rfind('}')
        
        if last_brace_idx == -1:
            # No closing brace found, just append
            return self.masked_test_code + f"\n{main_code}\n}}"
        
        # Insert main method before the last closing brace
        before_last_brace = self.masked_test_code[:last_brace_idx]
        after_last_brace = self.masked_test_code[last_brace_idx:]
        
        return before_last_brace + f"\n{main_code}\n" + after_last_brace
    
    def execute_test_case(self, test_code, compile_timeout=45, run_timeout=10):
        """Execute a single test case"""
        complete_java_code = ""
        try:
            # Extract main method body from test case
            # main_method_body = self.extract_main_method_body(test_code)
            # print(main_method_body)
            # Inject main method into masked test code
            complete_java_code = self.inject_main_code('\n' + test_code)
            # print(complete_java_code)
            # Write to temporary file and execute
            success, output = self._compile_and_run(complete_java_code, compile_timeout, run_timeout)
            return complete_java_code, success, output
            
        except Exception as e:
            return complete_java_code, False, f"Execution error: {e}"
    
    def _compile_and_run(self, java_code, compile_timeout, run_timeout):
        """Compile and run Java code in temporary directory"""
        # Use process-safe unique directory naming
        unique_id = str(uuid.uuid4())[:8]
        self.temp_dir = tempfile.mkdtemp(prefix=f"java_eval_{unique_id}_")
        
        try:
            # Write Java file with unique naming to avoid conflicts
            java_file = os.path.join(self.temp_dir, f"{self.cls_name}.java")
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(java_code)
            
            # Compile with optimizations for speed
            compile_result = subprocess.run(
                ['javac', '-J-XX:+TieredCompilation', '-J-XX:TieredStopAtLevel=1', java_file],
                capture_output=True,
                text=True,
                cwd=self.temp_dir,
                timeout=compile_timeout
            )
            
            if compile_result.returncode != 0:
                return False, f"Compilation failed: {compile_result.stderr}"
            
            # Run with assertions enabled
            run_result = subprocess.run(
                ['java', '-ea', self.cls_name],
                capture_output=True,
                text=True,
                cwd=self.temp_dir,
                timeout=run_timeout
            )
            
            if run_result.returncode != 0:
                return False, f"Runtime error: {run_result.stderr}"
            
            return True, run_result.stdout.strip()
            
        except subprocess.TimeoutExpired as e:
            return False, f"Timeout: {e}"
        except Exception as e:
            return False, f"Execution error: {e}"
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
                self.temp_dir = None
                
if __name__ == '__main__':
    code = "import java.io.*;\nimport java.util.*;\nimport java.text.*;\nimport java.math.*;\nimport java.util.regex.*;\npublic class Solution {\n    static int CeilIndex(int A[], int l, int r, int key) \n   {\n    int m;\n    while( r - l > 1 ) {\n        m = l + (r - l)/2;\n        if(A[m] >= key ) \n            r = m;\n        else\n            l = m;\n    }\n    return r;\n}\nstatic int f(int A[], int size) {\n    int[] tailTable   = new int[size];\n    int len; \n    for(int i=0;i<size;++i)\n        tailTable[i] = 0;\n    tailTable[0] = A[0];\n    len = 1;\n    for( int i = 1; i < size; i++ ) {\n        if( A[i] < tailTable[0] )\n            tailTable[0] = A[i];\n        else if( A[i] > tailTable[len-1] )\n            tailTable[len++] = A[i];\n        else\n            tailTable[CeilIndex(tailTable, -1, len-1, A[i])] = A[i];\n    }\n    return len;\n}\n\n}"
    exec = JavaExecutor(code, cls_name="Solution")
    main_code = "public static void main(String[] args) { int[] A = new int[]{42}; int size = 1; assert Solution.f(A, size) == 1; }"
    exec.execute_test_case(main_code)