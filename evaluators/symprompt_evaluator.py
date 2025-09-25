"""
Symprompt Evaluator

Standalone evaluator for unit test generation tasks on the Symprompt dataset.
Evaluates generated test code by measuring code coverage (line and branch coverage)
of focal methods in target repositories.

Supports:
- Code coverage evaluation with pytest
- Parallel processing for improved performance  
- Import fixing for local modules
- Line and branch coverage metrics
- Error handling and timeout management
"""

import os
import json
import re
import ast
import argparse
import tempfile
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

# Shared evaluation utilities
from evaluators.utils import get_optimal_worker_count
from evaluators.utils.python_version_manager import PythonVersionManager

# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "symprompt")

# Symprompt dataset specific paths - configured for TREAT setup
SYMPROMPT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "symprompt")
# SYMPROMPT_REPOS_DIR = "/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps"
SYMPROMPT_REPOS_DIR = os.path.join(SYMPROMPT_DATA_DIR, "test-apps")
SYMPROMPT_VENVS_DIR = os.path.join(PROJECT_ROOT, "virtual_environments", "symprompt")
FOCAL_METHODS_FILE = os.path.join(SYMPROMPT_DATA_DIR, "data", "focal_methods.jsonl")
FOCAL_METHODS_WITH_BRANCHES = os.path.join(
    SYMPROMPT_DATA_DIR, "data", "focal_methods_with_branches.jsonl"
)

# Python environment setup
python_manager = PythonVersionManager()
PYTHON_EXECUTABLE, setup_messages = python_manager.setup_environment()
print(setup_messages)

REPO_EXTRA_PIP_COMMANDS: Dict[str, List[str]] = {
    # "tqdm": [
    #     "pip install tqdm[notebook]",
    #     "pip install tqdm[rich]",
    #     "pip install ipywidgets",
    #     "pip install requests",
    # ],
}

REPO_TIMEOUTS: Dict[str, int] = {
    "youtube_dl": 30,
    "youtube": 30,
}

# ------------------ helpers: error classification & brief logs ------------------

def _classify_collection_error(stdout: str, stderr: str) -> Optional[str]:
    s = f"{stdout}\n{stderr}"
    if "Unknown config option:" in s:
        return "config_error"
    if "SyntaxError" in s or "ERROR collecting" in s:
        return "syntax_error" if "SyntaxError" in s else "collection_error"
    if "ModuleNotFoundError" in s or "ImportError" in s:
        return "import_error"
    if "No data was collected" in s:
        return "no_data_collected"
    return None

def _extract_syntax_brief(stdout: str, stderr: str) -> Dict[str, Any]:
    """从 pytest 报文中提取文件、行号和 SyntaxError 短消息。提取不到时返回最短兜底。"""
    s = f"{stdout}\n{stderr}"
    file_line = re.search(r'File "([^"]+)", line (\d+)', s)
    msg = None
    m = re.search(r"SyntaxError:\s*(.+)", s)
    if m:
        msg = m.group(1).strip()
    if not msg:
        m2 = re.search(r"^E\s+(.*SyntaxError.*)$", s, flags=re.M)
        if m2:
            msg = m2.group(1).strip()
    return {
        "type": "syntax_error",
        "file": file_line.group(1) if file_line else None,
        "line": int(file_line.group(2)) if file_line else None,
        "msg": (msg[:240] + "…") if msg and len(msg) > 240 else msg,
    }

def _brief_error(stdout: str, stderr: str, fallback_type: str = "error") -> Dict[str, Any]:
    """非语法错误的简要摘要（不回传整段日志）。"""
    s = (stderr or "").strip() or (stdout or "").strip()
    s = s.replace("\r", "")
    first_lines = "\n".join(s.splitlines()[:5])
    brief = first_lines[:400] + ("…" if len(first_lines) > 400 else "")
    return {"type": fallback_type, "msg": brief or None}

# --------------------------------- Evaluator ----------------------------------

class SympromptEvaluator:
    """Evaluator for Symprompt unit test generation tasks."""
    
    def __init__(self):
        self.focal_methods_data = self._load_focal_methods()
        self.branch_metadata = self._load_branch_metadata()
        self.repo_to_venv = self._get_repo_venv_mapping()
        self.prepared_repos = set()
        self.repo_preparation_lock = Lock()
        
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    def _load_focal_methods(self) -> Dict[str, tuple]:
        """Load focal methods data from the JSONL file."""
        focal_data = {}
        for data in self._load_jsonl(FOCAL_METHODS_FILE):
            project_name = data['project']
            module_name = data['module']
            class_name = data['class']
            method_name = data['method']
            key = f"{project_name}_{module_name}_{class_name}_{method_name}"
            focal_data[key] = (
                project_name,
                module_name,
                class_name,
                method_name,
                data['focal_method_lines'],
            )
        return focal_data

    def _load_branch_metadata(self) -> Dict[str, Dict[str, Any]]:
        data: Dict[str, Dict[str, Any]] = {}
        for row in self._load_jsonl(FOCAL_METHODS_WITH_BRANCHES):
            project_name = row['project']
            module_name = row['module']
            class_name = row['class']
            method_name = row['method']
            key = f"{project_name}_{module_name}_{class_name}_{method_name}"
            data[key] = row
        return data
    
    def _extract_key_from_ref_key(self, ref_key: str) -> str:
        """Extract the focal method key from ref_key format."""
        parts = ref_key.split('_')
        if len(parts) >= 5:
            key_parts = parts[2:-2]  # drop dataset & model_name & category/prompt_id
            return '_'.join(key_parts)
        return ref_key
    
    def _get_repo_venv_mapping(self) -> Dict[str, Path]:
        """Get mapping of repository names to their virtual environments."""
        repo_to_venv = {}
        if os.path.exists(SYMPROMPT_VENVS_DIR):
            venvs_dir = Path(SYMPROMPT_VENVS_DIR)
            for venv_path in venvs_dir.iterdir():
                if venv_path.is_dir() and venv_path.name.endswith("_env"):
                    repo_name = venv_path.name.replace("_env", "")
                    repo_to_venv[repo_name] = venv_path
        return repo_to_venv

    def _discover_requirement_files(self, repo_path: Path) -> List[Path]:
        """Locate requirement files within the repository."""
        requirement_files: List[Path] = []
        candidate_names = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
            "requirements-test.txt",
            "test_requirements.txt",
            "dev-requirements.txt",
        ]
        for name in candidate_names:
            path = repo_path / name
            if path.exists():
                requirement_files.append(path)
        req_dir = repo_path / "requirements"
        if req_dir.exists() and req_dir.is_dir():
            requirement_files.extend(sorted(req_dir.glob("*.txt")))
        return requirement_files

    def _ensure_repo_dependencies(self, repo_name: str, repo_path: Path, venv_path: Path) -> None:
        """Install repository dependencies into the associated virtual environment."""
        if repo_name in self.prepared_repos:
            return
        with self.repo_preparation_lock:
            if repo_name in self.prepared_repos:
                return

            pip_commands = [
                "pip install --upgrade --disable-pip-version-check pip setuptools wheel"
            ]
            pyproject_file = repo_path / "pyproject.toml"
            setup_file = repo_path / "setup.py"
            if pyproject_file.exists() or setup_file.exists():
                pip_commands.append("pip install -e .")
            for req_file in self._discover_requirement_files(repo_path):
                pip_commands.append(f'pip install -r "{req_file}"')
            extra_cmds = REPO_EXTRA_PIP_COMMANDS.get(repo_name, [])
            pip_commands.extend(extra_cmds)

            env_activate = f'source "{venv_path}/bin/activate"'
            install_failures = []
            for cmd in pip_commands:
                full_cmd = f'{env_activate} && cd "{repo_path}" && {cmd}'
                proc = subprocess.run(
                    full_cmd,
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    install_failures.append({
                        "command": cmd,
                        "returncode": proc.returncode,
                        "stdout": proc.stdout.strip(),
                        "stderr": proc.stderr.strip(),
                    })

            if install_failures:
                self.prepared_repos.add(repo_name)  # prevent retries each run
                failure_path = repo_path / "_symprompt_dependency_failures.json"
                try:
                    failure_path.write_text(json.dumps(install_failures, indent=2), encoding="utf-8")
                except OSError:
                    pass
            else:
                self.prepared_repos.add(repo_name)

    # ------------------------------- Pytest runner -------------------------------

    def run_pytest_with_coverage(
        self,
        code: str,
        repo_name: str,
        module_name: str,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Run pytest; try (1) disable plugins + minimal config, then (2) enable plugins + repo config.
        Return ONLY the 'better' attempt's stdout/stderr (prefer one that produced a valid XML)."""
        if repo_name not in self.repo_to_venv:
            return {"error": f"No virtual environment found for repository: {repo_name}"}
        
        venv_path = self.repo_to_venv[repo_name]
        repo_path = Path(SYMPROMPT_REPOS_DIR) / repo_name
        if not repo_path.exists():
            return {"error": f"Repository path not found: {repo_path}"}

        self._ensure_repo_dependencies(repo_name, repo_path, venv_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="test_",
                                        dir=repo_path, delete=False) as tf:
            tf.write(code)
            tmp_test_path = tf.name
            tmp_test_name = Path(tmp_test_path).name

        run_id = uuid.uuid4().hex
        artifact_dir = repo_path / "_symprompt_tmp_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        coverage_xml = artifact_dir / f"coverage_{run_id}.xml"
        coverage_data_file = artifact_dir / f".coverage_{run_id}"
        coverage_rcfile = artifact_dir / f"coverage_{run_id}.rc"
        coverage_rcfile.write_text("[run]\nbranch = True\n", encoding="utf-8")

        # minimal pytest ini for isolated first run
        pytest_min_ini = artifact_dir / "pytest_min.ini"
        try:
            pytest_min_ini.write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
        except OSError:
            pass

        timeout_seconds = REPO_TIMEOUTS.get(repo_name, 30)

        def _run_cmd(disable_plugins: bool, use_repo_config: bool, explicit_plugins: Optional[List[str]] = None):
            plugin_env = "export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && " if disable_plugins else "unset PYTEST_DISABLE_PLUGIN_AUTOLOAD && "
            ini_arg = "" if use_repo_config else f'-c "{pytest_min_ini}"'
            plug_arg = " ".join(f"-p {p}" for p in (explicit_plugins or []))
            bash_cmd = f'''
            source "{venv_path}/bin/activate" && \
            cd "{repo_path}" && \
            export PYTHONPATH=$(pwd) && \
            {plugin_env} \
            export COVERAGE_FILE="{coverage_data_file}" && \
            export COVERAGE_RCFILE="{coverage_rcfile}" && \
            coverage erase && \
            coverage run --branch -m pytest "{tmp_test_name}" {ini_arg} {plug_arg} \
              --disable-warnings \
              --continue-on-collection-errors \
              --ignore=tests || true && \
            coverage xml -o "{coverage_xml}" || true
            '''
            return subprocess.run(
                bash_cmd, shell=True, executable="/bin/bash",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_seconds, check=False,
            )

        try:
            # 1) 禁插件 + 最小 ini（隔离项目配置/插件噪声）
            proc1 = _run_cmd(disable_plugins=True, use_repo_config=False)
            err1 = _classify_collection_error(proc1.stdout, proc1.stderr)
            has_xml1 = coverage_xml.exists()

            best_proc, best_err = proc1, err1
            best_has_xml = has_xml1

            # 2) 如遇配置错误，则启插件 + 使用仓库配置重试
            if err1 == "config_error" or (not has_xml1):
                # 第二次尝试：允许 autoload，尊重仓库 setup.cfg/pytest.ini
                proc2 = _run_cmd(disable_plugins=False, use_repo_config=True)
                err2 = _classify_collection_error(proc2.stdout, proc2.stderr)
                has_xml2 = coverage_xml.exists()

                # 选择更好的一次：优先有 XML；否则选择错误等级更轻的
                def _rank(e: Optional[str]) -> int:
                    # 小数字更好
                    order = {
                        None: 0,
                        "no_data_collected": 1,
                        "import_error": 2,
                        "collection_error": 3,
                        "config_error": 4,
                        "syntax_error": 5,
                        "error": 6,
                    }
                    return order.get(e, 9)

                if (has_xml2 and not has_xml1) or (has_xml2 and _rank(err2) <= _rank(err1)):
                    best_proc, best_err, best_has_xml = proc2, err2, has_xml2

            result = {
                "exit_code": best_proc.returncode,
                "stdout": best_proc.stdout,
                "stderr": best_proc.stderr,
                "collection_error": best_err,
                "coverage_xml_path": str(coverage_xml) if best_has_xml else None,
                "artifact_dir": str(artifact_dir),
            }

            # 若有 XML，校验可解析（失败则视为解析错误）
            if best_has_xml:
                try:
                    ET.parse(coverage_xml)
                except Exception as exc:
                    result["coverage_error"] = f"Failed to parse coverage XML: {exc}"
                    result["coverage_xml_path"] = None

            return result

        except subprocess.TimeoutExpired:
            return {"error": "pytest execution timed out"}
        finally:
            try:
                os.unlink(tmp_test_path)
                if coverage_data_file.exists():
                    coverage_data_file.unlink()
                if coverage_rcfile.exists():
                    coverage_rcfile.unlink()
            except OSError:
                pass

    # ------------------------------- Coverage parse -------------------------------

    def calculate_method_coverage(
        self,
        module_name: str,
        start_line: int,
        end_line: int,
        coverage_xml_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate coverage metrics for a specific method using coverage XML."""
        if coverage_xml_path and os.path.exists(coverage_xml_path):
            try:
                return self._calculate_method_coverage_from_xml(
                    coverage_xml_path, module_name, start_line, end_line
                )
            except Exception as exc:
                return {"error": f"Failed to parse coverage XML: {exc}"}
        return {"error": "Coverage XML not produced"}

    def _calculate_method_coverage_from_xml(self, coverage_xml_path: str, module_name: str,
                                             start_line: int, end_line: int) -> Dict[str, Any]:
        tree = ET.parse(coverage_xml_path)
        root = tree.getroot()
        module_keywords = module_name.split(".")
        target_class = None

        for class_node in root.iter("class"):
            filename = class_node.get("filename", "")
            if all(keyword in filename for keyword in module_keywords):
                target_class = class_node
                break

        if target_class is None:
            module_basename = module_keywords[-1] if module_keywords else module_name
            candidate = f"{module_basename}.py"
            for class_node in root.iter("class"):
                filename = class_node.get("filename", "")
                if filename.endswith(candidate):
                    target_class = class_node
                    break

        if target_class is None:
            return {
                "line_coverage": 0,
                "branch_coverage": 0,
                "lines_covered": 0,
                "lines_total": 0,
                "branches_covered": 0,
                "branches_total": -1,
                "warning": f"Module {module_name} missing from coverage XML",
            }

        lines_node = target_class.find("lines")
        if lines_node is None:
            return {"error": "Coverage XML missing line data"}

        method_lines = set(range(start_line + 1, end_line + 2))
        covered_lines = set()
        relevant_lines = set()
        branches_total = 0
        branches_covered = 0

        for line_node in lines_node.iter("line"):
            try:
                line_number = int(line_node.get("number"))
            except (TypeError, ValueError):
                continue

            if line_number not in method_lines:
                continue

            relevant_lines.add(line_number)
            hits = int(line_node.get("hits", "0"))
            if hits > 0:
                covered_lines.add(line_number)

            if line_node.get("branch") == "true":
                condition = line_node.get("condition-coverage", "")
                if "(" in condition and "/" in condition:
                    try:
                        fraction = condition.split("(", 1)[1].split(")", 1)[0]
                        covered, total = fraction.split("/")
                        branches_covered += int(covered.strip())
                        branches_total += int(total.strip())
                    except (ValueError, IndexError):
                        continue

        line_coverage = len(covered_lines) / len(relevant_lines) * 100 if relevant_lines else 0
        if branches_total > 0:
            branch_coverage = branches_covered / branches_total * 100
        else:
            branches_total = -1
            branch_coverage = 0

        warning = None
        if not relevant_lines and branches_total == -1:
            warning = "Fallback import executed; focal lines not observed"

        return {
            "line_coverage": round(line_coverage, 2),
            "branch_coverage": round(branch_coverage, 2),
            "lines_covered": len(covered_lines),
            "lines_total": len(relevant_lines),
            "branches_covered": branches_covered,
            "branches_total": branches_total,
            **({"warning": warning} if warning else {}),
        }

    # ------------------------------- Per-item eval -------------------------------

    def evaluate_single_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single symprompt item (XML-only coverage), per-response metrics with compact logs."""
        try:
            if data.get("dataset") != "symprompt":
                return data

            ref_key = data.get("key") or data.get("ref_key", "")
            key = self._extract_key_from_ref_key(ref_key)
            if key not in self.focal_methods_data:
                data["evaluation_error"] = f"Focal method data not found for key: {key} (from ref_key: {ref_key})"
                return data

            project_name, module_name, _, _, method_lines = self.focal_methods_data[key]
            branch_meta: Optional[Dict[str, Any]] = self.branch_metadata.get(key)
            start_line, end_line = method_lines
            expected_has_branch = branch_meta.get("has_branch") if branch_meta else None
            expected_branches_total = branch_meta.get("branches_total") if branch_meta else None

            raw_responses = data.get("parsed_response", [])
            per_response_metrics: List[Dict[str, Any]] = []
            compact_logs: List[Dict[str, Any]] = []

            for response in raw_responses:
                test_code = response
                # 初始指标（占位）
                metrics = {
                    "csr": 0,
                    "line_coverage": 0,
                    "branch_coverage": 0 if (expected_has_branch is True or expected_has_branch is None) else -1,
                }
                if expected_has_branch is not None:
                    metrics["expected_has_branch"] = expected_has_branch
                if expected_branches_total is not None:
                    metrics["expected_branches_total"] = expected_branches_total

                # 运行 pytest + coverage
                pytest_result = self.run_pytest_with_coverage(
                    test_code, project_name, module_name, force_fallback=False
                )

                if "error" in pytest_result:
                    compact_logs.append({"ok": False, "type": "runner_error", "msg": pytest_result["error"]})
                    per_response_metrics.append(metrics)
                    continue

                stdout = pytest_result.get("stdout", "")
                stderr = pytest_result.get("stderr", "")
                cov_xml_path = pytest_result.get("coverage_xml_path")

                # 仅当没有 XML 时，才按分类输出失败简报
                if not cov_xml_path:
                    ctype = pytest_result.get("collection_error") or _classify_collection_error(stdout, stderr) or "no_coverage_xml"
                    if ctype == "syntax_error":
                        compact_logs.append({"ok": False, **_extract_syntax_brief(stdout, stderr)})
                    else:
                        compact_logs.append({"ok": False, **_brief_error(stdout, stderr, ctype)})
                    per_response_metrics.append(metrics)
                    # 清理
                    artifact_dir_path = pytest_result.get("artifact_dir")
                    if artifact_dir_path and os.path.isdir(artifact_dir_path):
                        try: os.rmdir(artifact_dir_path)
                        except OSError: pass
                    continue

                # 2) 有 XML：正常解析覆盖率
                coverage_metrics = self.calculate_method_coverage(
                    module_name, start_line, end_line, coverage_xml_path=cov_xml_path
                )

                # 清理
                try:
                    if cov_xml_path and os.path.exists(cov_xml_path):
                        os.unlink(cov_xml_path)
                except OSError:
                    pass
                artifact_dir_path = pytest_result.get("artifact_dir")
                if artifact_dir_path and os.path.isdir(artifact_dir_path):
                    try: os.rmdir(artifact_dir_path)
                    except OSError: pass

                if "error" in coverage_metrics:
                    compact_logs.append({"ok": False, **_brief_error(stdout, stderr, "coverage_parse_error")})
                    per_response_metrics.append(metrics)
                    continue

                # 成功路径：填充指标 + 极简日志
                metrics["csr"] = 1 if pytest_result.get("exit_code") not in [2, 3, 4] else 0
                line_cov = float(coverage_metrics.get("line_coverage", 0) or 0)
                metrics["line_coverage"] = float(round(line_cov, 2))

                # Preserve detailed coverage metrics
                metrics["cover_lines"] = coverage_metrics.get("lines_covered", 0)
                metrics["total_lines"] = coverage_metrics.get("lines_total", 0)
                metrics["cover_branches"] = coverage_metrics.get("branches_covered", 0)
                metrics["total_branches"] = coverage_metrics.get("branches_total", -1)

                branches_total_val = coverage_metrics.get("branches_total", -1)
                branch_cov_val = float(coverage_metrics.get("branch_coverage", 0) or 0)

                if expected_has_branch is True:
                    # 有预期：直接按观测值；即使没观测到分支也记 0（不再写 -1）
                    metrics["branch_coverage"] = round(branch_cov_val, 2)
                elif expected_has_branch is False:
                    # 明确“无预期”：-1（从汇总分母中剔除）
                    metrics["branch_coverage"] = -1
                else:
                    # 预期未知：根据 XML 观测推断
                    if isinstance(branches_total_val, (int, float)) and branches_total_val > 0:
                        metrics["branch_coverage"] = round(branch_cov_val, 2)
                        metrics["expected_has_branch"] = True
                    else:
                        metrics["branch_coverage"] = -1
                        metrics["expected_has_branch"] = False

                per_response_metrics.append(metrics)
                compact_logs.append({"ok": True})

            # 输出：每个 response 的 metrics；evaluation 为精简日志
            data["metrics"] = per_response_metrics
            data["evaluation"] = compact_logs
            return data

        except Exception as e:
            data["evaluation_error"] = f"Evaluation failed: {str(e)}"
            return data

# ------------------------------ Parallel + file IO ------------------------------

def _process_symprompt_item(args):
    """Process a single symprompt item - for parallel processing."""
    data, evaluator = args
    return evaluator.evaluate_single_item(data)

def evaluate_symprompt_file(
    filename: str,
    num_workers: int = None,
    run_error_items: bool = False,
) -> str:
    """Evaluate a symprompt prediction file and save results."""
    print(f"Evaluating symprompt file: {filename}")
    
    # Load data
    symprompt_data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            data = json.loads(line)
            if data.get("dataset") == "symprompt":  # Only process symprompt items
                symprompt_data.append(data)

    if not symprompt_data:
        print("No symprompt data found in file")
        return filename

    file_dir = os.path.dirname(filename)
    parent_dir = os.path.dirname(file_dir)
    evaluation_dir = os.path.join(parent_dir, 'evaluations')
    os.makedirs(evaluation_dir, exist_ok=True)

    base_filename = os.path.basename(filename)
    evaluation_file = os.path.join(evaluation_dir, base_filename)

    existing_results_by_key: Dict[str, Dict[str, Any]] = {}
    keys_to_evaluate: Optional[Set[str]] = None

    if run_error_items and os.path.exists(evaluation_file):
        keys_to_evaluate = set()
        error_keys = set()
        with open(evaluation_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = existing.get("ref_key") or existing.get("key")
                if not key:
                    continue
                existing_results_by_key[key] = existing
                evaluations = existing.get("evaluation")
                # 认为“需要重跑”的条件：没有 evaluation；或任何一条 ok=False；或旧结构含 error
                if not evaluations:
                    error_keys.add(key)
                    continue
                if any((isinstance(entry, dict) and (entry.get("ok") is False or entry.get("error"))) for entry in evaluations):
                    error_keys.add(key)

        if not error_keys:
            if existing_results_by_key:
                print("No prior error items detected in existing evaluation")

        missing_keys = set()
        for item in symprompt_data:
            key = item.get("ref_key") or item.get("key")
            if key and key not in existing_results_by_key:
                missing_keys.add(key)

        keys_to_evaluate = (error_keys or set()).union(missing_keys)

        if keys_to_evaluate:
            print(f"Re-evaluating {len(keys_to_evaluate)} items (errors or missing results)")
        else:
            print("No items require re-evaluation; returning existing results")
            return evaluation_file

    print(f"Found {len(symprompt_data)} symprompt items to evaluate")

    # Set up evaluator
    evaluator = SympromptEvaluator()

    # Determine optimal worker count
    if num_workers is None:
        num_workers = min(4, get_optimal_worker_count(len(symprompt_data)))  # Limit to 4 for coverage testing

    print(f"Using {num_workers} workers for evaluation")

    items_to_process = symprompt_data
    if keys_to_evaluate is not None:
        items_to_process = [
            item for item in symprompt_data
            if (item.get("ref_key") or item.get("key")) in keys_to_evaluate
        ]

        if not items_to_process:
            print("No matching items found for re-evaluation targets")
            if os.path.exists(evaluation_file):
                return evaluation_file
    elif run_error_items:
        print("[info] No previous evaluation file found; evaluating all items")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        item_args = [(data, evaluator) for data in items_to_process]
        evaluated_subset = list(tqdm(
            executor.map(_process_symprompt_item, item_args),
            total=len(item_args),
            desc="Evaluating symprompt items"
        ))

    evaluated_map = {}
    for item in evaluated_subset:
        key = item.get("ref_key") or item.get("key")
        if key:
            evaluated_map[key] = item

    if keys_to_evaluate is None:
        evaluated_data = evaluated_subset
    else:
        evaluated_data = []
        for item in symprompt_data:
            key = item.get("ref_key") or item.get("key")
            if key in evaluated_map:
                evaluated_data.append(evaluated_map[key])
            elif key and key in existing_results_by_key:
                evaluated_data.append(existing_results_by_key[key])
            else:
                evaluated_data.append(item)

    with open(evaluation_file, "w", encoding="utf-8") as f:
        for data in evaluated_data:
            f.write(json.dumps(data) + '\n')

    print(f"Evaluation completed. Results saved to: {evaluation_file}")
    return evaluation_file

# --------------------------- Branch metadata sync ---------------------------

def sync_evaluation_branch_metadata(
    evaluation_file: str,
    focal_metadata_path: str = FOCAL_METHODS_WITH_BRANCHES,
) -> Dict[str, Any]:
    """Update evaluation records with expected branch metadata and report mismatches."""
    evaluator = SympromptEvaluator()
    if focal_metadata_path != FOCAL_METHODS_WITH_BRANCHES:
        evaluator.branch_metadata = {
            f"{row['project']}_{row['module']}_{row['class']}_{row['method']}": row
            for row in evaluator._load_jsonl(focal_metadata_path)
        }

    processed = 0
    updated = 0
    mismatches: List[Dict[str, Any]] = []
    missing_metadata: List[str] = []
    updated_records: List[Dict[str, Any]] = []

    with open(evaluation_file, "r", encoding="utf-8") as fh:
        for line in fh:
            processed += 1
            record = json.loads(line)
            key = record.get("ref_key") or record.get("key")
            metadata = evaluator.branch_metadata.get(key) if key else None

            expected_has_branch = None
            expected_total = None
            if metadata:
                expected_has_branch = metadata.get("has_branch")
                expected_total = metadata.get("branches_total")
            elif key:
                missing_metadata.append(key)

            evaluations = record.get("evaluation")
            if (
                key
                and evaluations
                and isinstance(evaluations, list)
                and evaluations
                and isinstance(evaluations[0], dict)
            ):
                entry = evaluations[0]
                changed = False

                if expected_has_branch is not None and entry.get("expected_has_branch") != expected_has_branch:
                    entry["expected_has_branch"] = expected_has_branch
                    changed = True

                if expected_total is not None:
                    if entry.get("expected_branches_total") != expected_total:
                        entry["expected_branches_total"] = expected_total
                        changed = True

                    actual_total = entry.get("branches_total")
                    actual_total_int = None
                    if isinstance(actual_total, (int, float)):
                        actual_total_int = int(actual_total)

                    mismatch = False
                    if expected_total:
                        if actual_total_int is None or actual_total_int <= 0:
                            mismatch = True
                        elif actual_total_int != expected_total:
                            mismatch = True
                    else:
                        if actual_total_int and actual_total_int > 0:
                            mismatch = True

                    if mismatch:
                        mismatches.append(
                            {
                                "key": key,
                                "expected": expected_total,
                                "actual": actual_total,
                            }
                        )
                else:
                    if "expected_branches_total" in entry:
                        entry.pop("expected_branches_total", None)
                        changed = True

                metrics_list = record.get("metrics")
                if (
                    key and metrics_list and isinstance(metrics_list, list) and metrics_list
                    and isinstance(metrics_list[0], dict)
                ):
                    entry = metrics_list[0]
                    changed = False

                    if expected_has_branch is not None and entry.get("expected_has_branch") != expected_has_branch:
                        entry["expected_has_branch"] = expected_has_branch
                        changed = True

                    if expected_total is not None:
                        if entry.get("expected_branches_total") != expected_total:
                            entry["expected_branches_total"] = expected_total
                            changed = True

                    if changed:
                        metrics_list[0] = entry
                        record["metrics"] = metrics_list
                        updated += 1
            updated_records.append(record)

    with open(evaluation_file, "w", encoding="utf-8") as fh:
        for record in updated_records:
            fh.write(json.dumps(record) + "\n")

    return {
        "processed": processed,
        "updated": updated,
        "missing_metadata": sorted(set(missing_metadata)),
        "mismatches": mismatches,
    }

# ------------------------------- Dataset metrics -------------------------------

def compute_symprompt_metrics(filename: str) -> Dict[str, float]:
    """Compute overall metrics from evaluated symprompt file.
    读取 data["metrics"]（列表）——默认取每个 item 的首个 response 作为代表进行汇总。"""
    def _coerce_number(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    metrics = {
        "compilation_success_rate": 0,
        "average_line_coverage": 0,
        "average_branch_coverage": 0,
        "total_items": 0,
        "successful_items": 0,
        "csr": 0,
        "line_coverage": 0,
        "branch_coverage": -1
    }
    
    total_items = 0
    successful_items = 0
    total_line_coverage = 0
    total_branch_coverage = 0
    branch_items = 0  # number of items expected to have branches

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 仅统计有 metrics 的样本
            if "metrics" not in data or not isinstance(data["metrics"], list) or not data["metrics"]:
                continue

            total_items += 1
            m0 = data["metrics"][0]  # 默认取首个 response 的指标作为代表

            # CSR
            success_flag = bool(m0.get("csr"))
            if success_flag:
                successful_items += 1

            # Line coverage
            line_cov_value = _coerce_number(m0.get("line_coverage"))
            total_line_coverage += (line_cov_value or 0)

            # Branch coverage（仅统计“应有分支”的样本）
            expected_has_branch = m0.get("expected_has_branch")
            if expected_has_branch is None:
                expected_branches_total = _coerce_number(m0.get("expected_branches_total"))
                expected_has_branch = bool(expected_branches_total and expected_branches_total > 0)

            if expected_has_branch:
                branch_cov_value = _coerce_number(m0.get("branch_coverage"))
                if branch_cov_value is not None and branch_cov_value >= 0:
                    branch_items += 1
                    total_branch_coverage += branch_cov_value

    if total_items > 0:
        csr = successful_items / total_items * 100
        metrics["compilation_success_rate"] = csr
        metrics["total_items"] = total_items
        metrics["successful_items"] = successful_items
        metrics["csr"] = csr

        average_line = total_line_coverage / total_items
        metrics["average_line_coverage"] = average_line
        metrics["line_coverage"] = average_line

        if branch_items > 0:
            average_branch = total_branch_coverage / branch_items
        else:
            average_branch = -1
        metrics["average_branch_coverage"] = average_branch
        metrics["branch_coverage"] = average_branch
    else:
        metrics["branch_coverage"] = -1
    
    return metrics

# ----------------------------------- CLI -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symprompt unit test generations")
    parser.add_argument("input_file", help="JSONL file containing Symprompt predictions")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads (defaults to heuristic)",
    )
    parser.add_argument(
        "--run-error-items",
        action="store_true",
        help="Re-evaluate only items whose previous evaluation recorded an error",
    )
    parser.add_argument(
        "--sync-branches",
        action="store_true",
        help="After evaluation, align branch metadata with focal methods",
    )

    args = parser.parse_args()

    evaluation_file = evaluate_symprompt_file(
        args.input_file,
        num_workers=args.num_workers,
        run_error_items=args.run_error_items,
    )

    if args.sync_branches:
        summary = sync_evaluation_branch_metadata(evaluation_file)
        print(
            "\nBranch metadata sync summary:",
            f"processed={summary.get('processed')}, updated={summary.get('updated')}"
        )
        missing = summary.get("missing_metadata")
        if missing:
            print("Missing metadata for keys:", ", ".join(missing))
        mismatches = summary.get("mismatches", [])
        if mismatches:
            print("Branch total mismatches detected:")
            for entry in mismatches:
                print(f"  - {entry['key']}: expected={entry['expected']} actual={entry['actual']}")
        else:
            print("No branch total mismatches detected.")

    metrics = compute_symprompt_metrics(evaluation_file)

    print("\nEvaluation Metrics:")
    print(f"Total items: {metrics['total_items']}")
    print(f"Successful items: {metrics['successful_items']}")
    print(f"Compilation success rate: {metrics['compilation_success_rate']:.2f}%")
    print(f"Average line coverage: {metrics['average_line_coverage']:.2f}%")
    print(f"Average branch coverage: {metrics['average_branch_coverage']:.2f}%")
