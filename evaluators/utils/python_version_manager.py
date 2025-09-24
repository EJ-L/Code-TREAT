"""
Python Version Manager for PolyHumanEval

Handles Python version detection, validation, and provides setup instructions
for ensuring Python 3.9 is available for PolyHumanEval evaluations.
"""

import os
import sys
import subprocess
import shutil
from typing import Optional, Tuple, List
from pathlib import Path


class PythonVersionManager:
    """Manages Python version detection and setup for PolyHumanEval."""
    
    RECOMMENDED_VERSION = "3.9"
    ACCEPTABLE_VERSIONS = ["3.9", "3.10", "3.11"]  # Versions that work well
    
    @staticmethod
    def get_venv_python() -> Optional[str]:
        """Get Python executable from the local PolyHumanEval venv."""
        # Get the project root directory (parent of evaluators)
        project_root = Path(__file__).parent.parent.parent
        venv_path = project_root / "virtual_environments" / "polyhumaneval" / "venv" / "python3.9"
        
        if not venv_path.exists():
            return None
            
        # Get Python executable from venv
        if sys.platform.startswith("win"):
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            # Verify it's actually Python 3.9
            try:
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version_output = result.stdout + result.stderr
                    if "Python 3.9" in version_output:
                        return str(python_exe)
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def find_python_executable(version: str = "3.9") -> Optional[str]:
        """
        Find Python executable for a specific version.
        
        Args:
            version: Python version to find (e.g., "3.9")
            
        Returns:
            Path to Python executable or None if not found
        """
        candidates = [
            f"python{version}",
            f"python{version[0]}",  # Try python3
            "python",
            sys.executable,  # Current Python
        ]
        
        # Add common installation paths
        if sys.platform == "darwin":  # macOS
            candidates.extend([
                f"/usr/local/bin/python{version}",
                f"/opt/homebrew/bin/python{version}",
                f"/usr/bin/python{version}",
                f"~/.pyenv/versions/{version}*/bin/python",
            ])
        elif sys.platform.startswith("linux"):
            candidates.extend([
                f"/usr/bin/python{version}",
                f"/usr/local/bin/python{version}",
                f"~/.pyenv/versions/{version}*/bin/python",
            ])
        
        for candidate in candidates:
            candidate = os.path.expanduser(candidate)
            if "*" in candidate:
                # Handle glob patterns
                import glob
                matches = glob.glob(candidate)
                if matches:
                    candidate = matches[0]
                else:
                    continue
            
            if shutil.which(candidate):
                # Verify it's actually the right version
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
    def get_python_version(executable: str) -> Optional[str]:
        """Get the version of a Python executable."""
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_output = result.stdout + result.stderr
                # Extract version number (e.g., "Python 3.9.7" -> "3.9")
                import re
                match = re.search(r"Python (\d+\.\d+)", version_output)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return None
    
    @staticmethod
    def check_python_availability() -> Tuple[bool, str, List[str]]:
        """
        Check Python availability and provide recommendations.
        
        Returns:
            Tuple of (is_available, executable_path, messages)
        """
        messages = []
        
        # First, check for local PolyHumanEval venv
        venv_python = PythonVersionManager.get_venv_python()
        if venv_python:
            messages.append(f"✓ Found PolyHumanEval venv Python 3.9 at: {venv_python}")
            messages.append("  (Using dedicated virtual environment)")
            return True, venv_python, messages
        
        # Then, try to find system Python 3.9
        python_exe = PythonVersionManager.find_python_executable("3.9")
        if python_exe:
            messages.append(f"✓ Found system Python 3.9 at: {python_exe}")
            messages.append("  (Consider creating dedicated venv with: python environment_setup/setup_polyhumaneval_environments.py)")
            return True, python_exe, messages
        
        # Try other acceptable versions
        for version in PythonVersionManager.ACCEPTABLE_VERSIONS:
            python_exe = PythonVersionManager.find_python_executable(version)
            if python_exe:
                messages.append(f"⚠ Found Python {version} at: {python_exe} (not ideal but acceptable)")
                return True, python_exe, messages
        
        # No suitable Python found
        messages.append("✗ Python 3.9 not found on your system")
        messages.append("")
        messages.append("🚀 Quick Setup (Recommended):")
        messages.append("   Run: python environment_setup/setup_polyhumaneval_environments.py")
        messages.append("   This will create a dedicated Python 3.9 environment for PolyHumanEval")
        messages.append("")
        messages.append("Or install Python 3.9 manually using one of the following methods:")
        messages.extend(PythonVersionManager.get_installation_instructions())
        
        return False, "", messages
    
    @staticmethod
    def get_installation_instructions() -> List[str]:
        """Get platform-specific installation instructions."""
        instructions = []
        
        if sys.platform == "darwin":  # macOS
            instructions.extend([
                "",
                "=== macOS Installation Options ===",
                "",
                "Option 1: Using Homebrew (recommended):",
                "  brew install python@3.9",
                "",
                "Option 2: Using pyenv:",
                "  brew install pyenv",
                "  pyenv install 3.9.18",
                "  pyenv global 3.9.18",
                "",
                "Option 3: Using official installer:",
                "  Download from https://www.python.org/downloads/release/python-3918/",
                "",
                "Option 4: Using conda/miniconda:",
                "  conda create -n py39 python=3.9",
                "  conda activate py39",
            ])
        
        elif sys.platform.startswith("linux"):
            instructions.extend([
                "",
                "=== Linux Installation Options ===",
                "",
                "Option 1: Using apt (Ubuntu/Debian):",
                "  sudo apt update",
                "  sudo apt install python3.9 python3.9-dev python3.9-venv",
                "",
                "Option 2: Using yum (RHEL/CentOS/Fedora):",
                "  sudo yum install python39",
                "",
                "Option 3: Using pyenv:",
                "  curl https://pyenv.run | bash",
                "  pyenv install 3.9.18",
                "  pyenv global 3.9.18",
                "",
                "Option 4: Build from source:",
                "  wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz",
                "  tar -xf Python-3.9.18.tgz",
                "  cd Python-3.9.18",
                "  ./configure --enable-optimizations",
                "  make -j$(nproc)",
                "  sudo make altinstall",
                "",
                "Option 5: Using conda/miniconda:",
                "  conda create -n py39 python=3.9",
                "  conda activate py39",
            ])
        
        elif sys.platform.startswith("win"):
            instructions.extend([
                "",
                "=== Windows Installation Options ===",
                "",
                "Option 1: Using official installer (recommended):",
                "  1. Download from https://www.python.org/downloads/release/python-3918/",
                "  2. Run the installer",
                "  3. Check 'Add Python to PATH'",
                "",
                "Option 2: Using Windows Store:",
                "  Search for 'Python 3.9' in Microsoft Store",
                "",
                "Option 3: Using Chocolatey:",
                "  choco install python39",
                "",
                "Option 4: Using conda/miniconda:",
                "  conda create -n py39 python=3.9",
                "  conda activate py39",
            ])
        
        instructions.extend([
            "",
            "=== Using uv (Universal Python packager) ===",
            "  uv python install 3.9",
            "  uv venv --python 3.9",
            "  source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate",
            "",
            "After installation, verify with:",
            "  python3.9 --version",
            "",
            "Then set the environment variable:",
            "  export POLYHUMANEVAL_PYTHON=python3.9",
            "",
        ])
        
        return instructions
    
    @staticmethod
    def setup_environment() -> Tuple[str, List[str]]:
        """
        Set up the Python environment for PolyHumanEval.
        
        Returns:
            Tuple of (python_executable, setup_messages)
        """
        messages = []
        
        # Check if environment variable is set
        env_python = os.environ.get('POLYHUMANEVAL_PYTHON')
        if env_python:
            if shutil.which(env_python):
                version = PythonVersionManager.get_python_version(env_python)
                messages.append(f"Using Python from POLYHUMANEVAL_PYTHON: {env_python} (version {version})")
                return env_python, messages
            else:
                messages.append(f"Warning: POLYHUMANEVAL_PYTHON={env_python} not found, searching for alternatives...")
        
        # Check for local venv first (highest priority)
        venv_python = PythonVersionManager.get_venv_python()
        if venv_python:
            messages.append(f"✓ Using PolyHumanEval dedicated environment: {venv_python}")
            return venv_python, messages
        
        # Auto-detect system Python
        available, python_exe, check_messages = PythonVersionManager.check_python_availability()
        messages.extend(check_messages)
        
        if available:
            messages.append(f"\nUsing Python executable: {python_exe}")
            messages.append(f"To make this permanent, add to your shell config:")
            messages.append(f"  export POLYHUMANEVAL_PYTHON={python_exe}")
            return python_exe, messages
        else:
            # Fallback to current Python with warning
            messages.append(f"\n⚠ WARNING: Falling back to current Python ({sys.version})")
            messages.append("This may cause compatibility issues with PolyHumanEval!")
            return sys.executable, messages


def print_setup_guide():
    """Print a comprehensive setup guide."""
    print("\n" + "=" * 60)
    print("PolyHumanEval Python Setup Guide")
    print("=" * 60)
    
    manager = PythonVersionManager()
    python_exe, messages = manager.setup_environment()
    
    for msg in messages:
        print(msg)
    
    print("\n" + "=" * 60)
    print(f"Final Python executable: {python_exe}")
    version = manager.get_python_version(python_exe)
    print(f"Python version: {version}")
    
    if version and not version.startswith("3.9"):
        print("\n⚠ WARNING: Not using Python 3.9 - some tests may fail!")
    
    print("=" * 60 + "\n")
    
    return python_exe


if __name__ == "__main__":
    # Run setup guide when executed directly
    print_setup_guide()