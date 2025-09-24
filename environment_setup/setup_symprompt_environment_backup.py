#!/usr/bin/env python3
"""
Symprompt Environment Setup Script

This script creates isolated virtual environments for each GitHub project in the 
Symprompt dataset. It uses the existing repositories from the RA_ARISE TREAT 
setup and creates dedicated venvs under virtual_environments/symprompt/.

Key Features:
- Creates isolated virtual environments for each Symprompt project
- Installs project dependencies and testing tools
- Uses existing test-apps from RA_ARISE setup temporarily
- Supports both Python 3.9 and fallback versions
- Validates installations with import tests

Author: TREAT Framework
Version: 1.0.0
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('symprompt_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
VIRTUAL_ENVS_DIR = PROJECT_ROOT / "virtual_environments" / "symprompt"
SYMPROMPT_DATA_FILE = PROJECT_ROOT / "data" / "symprompt" / "data" / "focal_methods.jsonl"

# Use existing RA_ARISE test-apps temporarily
RA_ARISE_TEST_APPS = Path("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps")
RA_ARISE_VENVS = Path("/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/venvs")

# Project name mappings based on RA_ARISE setup_venv.sh
PROJECT_NAME_DIR_MAP = {
    "apimd": "apimd",
    "codetiming": "codetiming", 
    "cookiecutter": "cookiecutter",
    "dataclasses_json": "dataclasses_json",
    "docstring_parser": "docstring_parser",
    "flutes": "flutes",
    "flutils": "flutils",
    "httpie": "httpie",
    "isort": "isort",
    "py_backwards": "py_backwards",
    "pymonet": "pymonet",
    "pypara": "pypara",
    "pysnooper": "pysnooper",
    "semantic_release": "semantic_release",
    "string_utils": "string_utils",
    "pytutils": "pytutils",
    "sanic": "sanic",
    "sty": "sty",
    "thefuck": "thefuck",
    "thonny": "thonny",
    "tornado": "tornado",
    "tqdm": "tqdm",
    "typesystem": "typesystem",
    "youtube_dl": "youtube_dl",
}

# Essential packages for testing
BASE_PACKAGES = [
    "requests", "psutil", "colorama", "decorator", "typed_ast", 
    "colorlog", "sanic_routing"
]

TEST_PACKAGES = [
    "pytest", "pytest-asyncio", "pytest-cov", "pytest-json-report", 
    "coverage", "unittest-xml-reporting"
]

class PythonVersionManager:
    """Manages Python version detection for Symprompt setup."""
    
    RECOMMENDED_VERSION = "3.9"
    ACCEPTABLE_VERSIONS = ["3.9", "3.10", "3.11"]
    
    @staticmethod
    def find_python_executable(version: str = "3.9") -> Optional[str]:
        """Find Python executable for a specific version."""
        candidates = [
            f"python{version}",
            f"python{version[0]}",
            "python",
            sys.executable,
        ]
        
        # Add common installation paths
        if sys.platform == "darwin":  # macOS
            candidates.extend([
                f"/usr/local/bin/python{version}",
                f"/opt/homebrew/bin/python{version}",
                f"/usr/bin/python{version}",
            ])
        elif sys.platform.startswith("linux"):
            candidates.extend([
                f"/usr/bin/python{version}",
                f"/usr/local/bin/python{version}",
            ])
        
        for candidate in candidates:
            if shutil.which(candidate):
                try:
                    result = subprocess.run(
                        [candidate, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version_output = result.stdout + result.stderr
                        if f"Python {version}" in version_output:
                            return candidate
                except Exception:
                    continue
        
        return None
    
    @staticmethod
    def setup_python() -> str:
        """Setup and return the best Python executable."""
        # First try Python 3.9
        python_exe = PythonVersionManager.find_python_executable("3.9")
        if python_exe:
            logger.info(f"✓ Found Python 3.9: {python_exe}")
            return python_exe
        
        # Try other acceptable versions
        for version in PythonVersionManager.ACCEPTABLE_VERSIONS:
            python_exe = PythonVersionManager.find_python_executable(version)
            if python_exe:
                logger.warning(f"⚠ Using Python {version}: {python_exe} (Python 3.9 preferred)")
                return python_exe
        
        # Fallback to current Python
        logger.warning(f"⚠ Falling back to current Python: {sys.executable}")
        return sys.executable

class SympromptEnvironmentManager:
    """Manages Symprompt virtual environments setup."""
    
    def __init__(self):
        self.python_exe = PythonVersionManager.setup_python()
        self.projects = self._load_projects_from_data()
        
    def _load_projects_from_data(self) -> List[str]:
        """Load unique project names from symprompt focal methods data."""
        if not SYMPROMPT_DATA_FILE.exists():
            logger.error(f"Symprompt data file not found: {SYMPROMPT_DATA_FILE}")
            return []
        
        projects = set()
        try:
            with open(SYMPROMPT_DATA_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        project = data.get("project")
                        if project:
                            projects.add(project)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading symprompt data: {e}")
            return []
        
        project_list = sorted(list(projects))
        logger.info(f"Found {len(project_list)} projects in symprompt dataset: {project_list}")
        return project_list
    
    def _check_ra_arise_setup(self) -> bool:
        """Check if RA_ARISE test-apps are available."""
        if not RA_ARISE_TEST_APPS.exists():
            logger.error(f"RA_ARISE test-apps not found at: {RA_ARISE_TEST_APPS}")
            logger.error("Please ensure the RA_ARISE setup is available.")
            return False
        
        # Check if we have the expected projects
        available_projects = [d.name for d in RA_ARISE_TEST_APPS.iterdir() if d.is_dir()]
        missing_projects = [p for p in self.projects if p not in available_projects]
        
        if missing_projects:
            logger.warning(f"Missing projects in RA_ARISE test-apps: {missing_projects}")
        
        logger.info(f"RA_ARISE test-apps found with {len(available_projects)} projects")
        return True
    
    def create_venv(self, project_name: str) -> bool:
        """Create virtual environment for a project."""
        venv_path = VIRTUAL_ENVS_DIR / f"{project_name}_env"
        
        if venv_path.exists():
            logger.info(f"Virtual environment already exists for {project_name}")
            return True
        
        logger.info(f"Creating virtual environment for {project_name}...")
        
        try:
            # Create venv
            subprocess.run(
                [self.python_exe, "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True
            )
            
            # Get venv python
            if sys.platform.startswith("win"):
                venv_python = venv_path / "Scripts" / "python.exe"
                venv_pip = venv_path / "Scripts" / "pip.exe"
            else:
                venv_python = venv_path / "bin" / "python"
                venv_pip = venv_path / "bin" / "pip"
            
            # Upgrade pip
            subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)
            
            logger.info(f"✓ Created virtual environment: {venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment for {project_name}: {e}")
            return False
    
    def install_project_dependencies(self, project_name: str) -> bool:
        """Install project dependencies in its virtual environment."""
        venv_path = VIRTUAL_ENVS_DIR / f"{project_name}_env"
        repo_path = RA_ARISE_TEST_APPS / PROJECT_NAME_DIR_MAP.get(project_name, project_name)
        
        if not venv_path.exists():
            logger.error(f"Virtual environment not found for {project_name}")
            return False
        
        if not repo_path.exists():
            logger.error(f"Repository not found for {project_name}: {repo_path}")
            return False
        
        # Get venv executables
        if sys.platform.startswith("win"):
            venv_pip = venv_path / "Scripts" / "pip.exe"
        else:
            venv_pip = venv_path / "bin" / "pip"
        
        logger.info(f"Installing dependencies for {project_name}...")
        
        try:
            # Install requirements.txt if exists
            requirements_file = repo_path / "requirements.txt"
            if requirements_file.exists():
                logger.info(f"  Installing from requirements.txt")
                subprocess.run([
                    str(venv_pip), "install", "-r", str(requirements_file)
                ], check=True, capture_output=True)
            
            # Install base packages
            for pkg in BASE_PACKAGES:
                try:
                    subprocess.run([
                        str(venv_pip), "show", pkg
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    logger.info(f"  Installing missing base package: {pkg}")
                    subprocess.run([
                        str(venv_pip), "install", pkg
                    ], check=True)
            
            # Install test packages
            for pkg in TEST_PACKAGES:
                try:
                    subprocess.run([
                        str(venv_pip), "show", pkg
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    logger.info(f"  Installing missing test package: {pkg}")
                    subprocess.run([
                        str(venv_pip), "install", pkg
                    ], check=True)
            
            # Install project in development mode (if applicable)
            setup_py = repo_path / "setup.py"
            pyproject_toml = repo_path / "pyproject.toml"
            
            if setup_py.exists() or pyproject_toml.exists():
                logger.info(f"  Installing {project_name} in development mode")
                if project_name == "sty":
                    # Special handling for sty (fixed pyproject.toml with build dependencies)
                    subprocess.run([
                        str(venv_pip), "install", "-e", str(repo_path)
                    ], check=True)
                elif project_name == "isort":
                    # Special handling for isort (fallback to regular install)
                    try:
                        subprocess.run([
                            str(venv_pip), "install", "-e", str(repo_path)
                        ], check=True)
                    except subprocess.CalledProcessError:
                        logger.warning(f"Editable install failed for {project_name}, trying regular install")
                        subprocess.run([
                            str(venv_pip), "install", str(repo_path)
                        ], check=True)
                else:
                    subprocess.run([
                        str(venv_pip), "install", "-e", str(repo_path)
                    ], check=True)
            
            logger.info(f"✓ Dependencies installed for {project_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies for {project_name}: {e}")
            return False
    
    def verify_installation(self, project_name: str) -> bool:
        """Verify project installation in its virtual environment."""
        venv_path = VIRTUAL_ENVS_DIR / f"{project_name}_env"
        
        if sys.platform.startswith("win"):
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"
        
        try:
            # Test import
            result = subprocess.run([
                str(venv_python), "-c", f"import {project_name}; print(f'✓ {project_name} imported successfully')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"✓ Verification passed for {project_name}")
                return True
            else:
                logger.warning(f"Import test failed for {project_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Verification failed for {project_name}: {e}")
            return False
    
    def setup_all_environments(self) -> Dict[str, bool]:
        """Setup virtual environments for all Symprompt projects."""
        logger.info("Starting Symprompt environment setup...")
        
        # Check RA_ARISE setup
        if not self._check_ra_arise_setup():
            return {}
        
        # Ensure virtual environments directory exists
        VIRTUAL_ENVS_DIR.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for project in self.projects:
            if project not in PROJECT_NAME_DIR_MAP:
                logger.warning(f"Project {project} not found in mapping, skipping")
                results[project] = False
                continue
            
            logger.info(f"\n→ Setting up environment for: {project}")
            
            # Create venv
            if not self.create_venv(project):
                results[project] = False
                continue
            
            # Install dependencies
            if not self.install_project_dependencies(project):
                results[project] = False
                continue
            
            # Verify installation
            if not self.verify_installation(project):
                results[project] = False
                continue
            
            results[project] = True
            logger.info(f"✓ Completed setup for {project}")
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print setup summary."""
        successful = [p for p, success in results.items() if success]
        failed = [p for p, success in results.items() if not success]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Symprompt Environment Setup Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total projects: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            logger.info(f"\n✓ Successful projects:")
            for project in successful:
                venv_path = VIRTUAL_ENVS_DIR / f"{project}_env"
                logger.info(f"  - {project} → {venv_path}")
        
        if failed:
            logger.info(f"\n✗ Failed projects:")
            for project in failed:
                logger.info(f"  - {project}")
        
        logger.info(f"\nVirtual environments created in: {VIRTUAL_ENVS_DIR}")
        logger.info(f"Setup log saved to: symprompt_setup.log")

def main():
    """Main setup entry point."""
    print(f"\n{'='*60}")
    print(f"Symprompt Environment Setup")
    print(f"{'='*60}")
    
    try:
        manager = SympromptEnvironmentManager()
        results = manager.setup_all_environments()
        manager.print_summary(results)
        
        # Exit with error code if any setup failed
        if any(not success for success in results.values()):
            sys.exit(1)
        
        logger.info(f"\n🎉 All Symprompt environments setup successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()