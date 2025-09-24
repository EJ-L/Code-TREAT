#!/usr/bin/env python3
import os, shutil, subprocess, tempfile, textwrap, traceback
import builtins, textwrap, traceback

class PythonExecutor:
    def __init__(self, masked_test_code, timeout_sec=5):
        self.masked_test_code = masked_test_code
        self.timeout_sec = timeout_sec

    def execute_test_case(self, test_code):
        complete_code = self.masked_test_code + "\n" + test_code
        return self._compile_and_run(complete_code)

    def _compile_and_run(self, python_code):
        # write code out to a temp file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            path = tf.name
            tf.write(textwrap.dedent(python_code))

        try:
            # run it in a subprocess with a timeout
            # Set PYTHONHASHSEED to make set/dict iteration deterministic
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '0'
            proc = subprocess.run(
                ["python3", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
                env=env
            )
            if proc.returncode == 0:
                return python_code, True, "OK"
            else:
                return python_code, False, proc.stderr or proc.stdout

        except subprocess.TimeoutExpired:
            return python_code, False, f"TimeoutExpired after {self.timeout_sec}s"

        finally:
            try:
                os.remove(path)
            except OSError:
                pass