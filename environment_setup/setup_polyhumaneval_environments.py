#!/usr/bin/env python3
"""
PolyHumanEval Environment Setup Script

Creates a dedicated Python 3.9 virtual environment for PolyHumanEval evaluations.
This script is flexible and works with any Python installation method:
- uv, pyenv, conda, system Python, etc.

Usage:
    python environment_setup/setup_polyhumaneval_environments.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List


class PolyHumanEvalEnvironmentSetup:
    """Setup and manage the PolyHumanEval Python 3.9 environment."""
    
    def __init__(self):
        # Get project root (parent of environment_setup directory)
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "virtual_environments" / "polyhumaneval" / "venv" / "python3.9"
        self.required_version = "3.9"
    
    def find_python39(self) -> Optional[str]:
        """Find a suitable Python 3.9 installation."""
        print("🔍 Searching for Python 3.9...")
        
        # Try different Python executables
        candidates = [
            "python3.9",
            "python3",
            "python",
        ]
        
        # Add platform-specific paths
        if sys.platform == "darwin":  # macOS
            candidates.extend([
                "/usr/local/bin/python3.9",
                "/opt/homebrew/bin/python3.9",
                "/usr/bin/python3.9",
            ])
        elif sys.platform.startswith("linux"):
            candidates.extend([
                "/usr/bin/python3.9",
                "/usr/local/bin/python3.9",
            ])
        
        # Check pyenv installations
        home = Path.home()
        pyenv_versions = home / ".pyenv" / "versions"
        if pyenv_versions.exists():
            for version_dir in pyenv_versions.iterdir():
                if version_dir.is_dir() and "3.9" in version_dir.name:
                    candidates.append(str(version_dir / "bin" / "python"))
        
        # Check uv managed Python
        uv_python_dir = home / ".local" / "share" / "uv" / "python"
        if uv_python_dir.exists():
            for python_dir in uv_python_dir.iterdir():
                if python_dir.is_dir() and "3.9" in python_dir.name:
                    # uv structure: ~/.local/share/uv/python/cpython-3.9.x-platform/bin/python
                    python_exe = python_dir / "bin" / "python"
                    if python_exe.exists():
                        candidates.append(str(python_exe))
        
        # Test candidates
        for candidate in candidates:
            try:
                if shutil.which(candidate):
                    result = subprocess.run(
                        [candidate, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version_output = result.stdout + result.stderr
                        if "Python 3.9" in version_output:
                            print(f"✓ Found Python 3.9: {candidate}")
                            return candidate
            except Exception:
                continue
        
        return None
    
    def suggest_python39_installation(self) -> List[str]:
        """Provide platform-specific installation suggestions."""
        suggestions = [
            "",
            "❌ Python 3.9 not found. Here are installation options:",
            "",
        ]
        
        if shutil.which("uv"):
            suggestions.extend([
                "📦 Using uv (recommended if you have it):",
                "   uv python install 3.9",
                "",
            ])
        
        if sys.platform == "darwin":  # macOS
            suggestions.extend([
                "🍎 macOS options:",
                "   1. Homebrew: brew install python@3.9",
                "   2. pyenv: brew install pyenv && pyenv install 3.9.18",
                "   3. Official: https://www.python.org/downloads/",
                "",
            ])
        
        elif sys.platform.startswith("linux"):
            suggestions.extend([
                "🐧 Linux options:",
                "   1. Ubuntu/Debian: sudo apt install python3.9 python3.9-venv",
                "   2. RHEL/Fedora: sudo yum install python39",
                "   3. pyenv: curl https://pyenv.run | bash && pyenv install 3.9.18",
                "",
            ])
        
        elif sys.platform.startswith("win"):
            suggestions.extend([
                "🪟 Windows options:",
                "   1. Official installer: https://www.python.org/downloads/",
                "   2. Windows Store: Search for 'Python 3.9'",
                "   3. Chocolatey: choco install python39",
                "",
            ])
        
        suggestions.extend([
            "🐍 Universal options:",
            "   1. conda: conda create -n py39 python=3.9",
            "   2. From source: Download and build from python.org",
            "",
            "After installation, run this script again:",
            f"   python {__file__}",
        ])
        
        return suggestions
    
    def create_venv(self, python_exe: str) -> bool:
        """Create the virtual environment."""
        print(f"🏗️  Creating virtual environment at: {self.venv_path}")
        
        try:
            # Remove existing venv if it exists
            if self.venv_path.exists():
                print("   Removing existing environment...")
                shutil.rmtree(self.venv_path)
            
            # Ensure parent directories exist
            self.venv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create new venv
            result = subprocess.run(
                [python_exe, "-m", "venv", str(self.venv_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to create virtual environment:")
                print(f"   {result.stderr}")
                return False
            
            print("✅ Virtual environment created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> Optional[str]:
        """Get the Python executable from the venv."""
        if sys.platform.startswith("win"):
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"
        
        return str(python_exe) if python_exe.exists() else None
    
    def verify_venv(self) -> bool:
        """Verify the virtual environment is working correctly."""
        print("🧪 Verifying virtual environment...")
        
        python_exe = self.get_venv_python()
        if not python_exe:
            print("❌ Virtual environment Python not found")
            return False
        
        try:
            # Test Python version
            result = subprocess.run(
                [python_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to run Python in venv: {result.stderr}")
                return False
            
            version_output = result.stdout + result.stderr
            if "Python 3.9" not in version_output:
                print(f"❌ Wrong Python version in venv: {version_output}")
                return False
            
            # Test basic imports
            test_code = "import sys, json, subprocess; print('Basic imports OK')"
            result = subprocess.run(
                [python_exe, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print(f"❌ Basic import test failed: {result.stderr}")
                return False
            
            print(f"✅ Virtual environment verified: {version_output.strip()}")
            return True
            
        except Exception as e:
            print(f"❌ Error verifying environment: {e}")
            return False
    
    def install_minimal_deps(self) -> bool:
        """Install minimal dependencies if needed."""
        print("📦 Checking for required packages...")
        
        python_exe = self.get_venv_python()
        if not python_exe:
            return False
        
        # Usually no extra deps needed for PolyHumanEval, but we can add here if needed
        # For now, just ensure pip is up to date
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ pip available in virtual environment")
                return True
            else:
                print("ℹ️  pip not available, but that's OK for PolyHumanEval")
                return True
                
        except Exception:
            print("ℹ️  Could not check pip, but that's OK for PolyHumanEval")
            return True
    
    def setup(self) -> bool:
        """Run the complete setup process."""
        print("=" * 70)
        print("🚀 PolyHumanEval Python 3.9 Environment Setup")
        print("=" * 70)
        print(f"📍 Target location: {self.venv_path}")
        print("")
        
        # Check if venv already exists and is valid
        if self.venv_path.exists():
            print("📁 Existing virtual environment found")
            if self.verify_venv():
                print("✅ Environment is already set up and working!")
                print(f"   Location: {self.venv_path}")
                print(f"   Python: {self.get_venv_python()}")
                return True
            else:
                print("🔧 Existing environment has issues, recreating...")
        
        # Find Python 3.9
        python_exe = self.find_python39()
        if not python_exe:
            suggestions = self.suggest_python39_installation()
            for suggestion in suggestions:
                print(suggestion)
            return False
        
        # Create virtual environment
        if not self.create_venv(python_exe):
            return False
        
        # Verify environment
        if not self.verify_venv():
            return False
        
        # Install minimal dependencies
        if not self.install_minimal_deps():
            return False
        
        print("")
        print("=" * 70)
        print("🎉 Setup completed successfully!")
        print(f"📍 Virtual environment: {self.venv_path}")
        print(f"🐍 Python executable: {self.get_venv_python()}")
        print("")
        print("The PolyHumanEval evaluator will now automatically use this environment.")
        print("You can continue using your main project with Python 3.12 via uv.")
        print("=" * 70)
        
        return True


def main():
    """Main setup function."""
    setup = PolyHumanEvalEnvironmentSetup()
    success = setup.setup()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())