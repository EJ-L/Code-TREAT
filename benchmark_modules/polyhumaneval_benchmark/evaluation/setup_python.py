#!/usr/bin/env python3
"""
Setup script to configure Python version for PolyHumanEval evaluation.
Run this script to check and configure Python 3.9 for the evaluation.
"""

import os
import sys
import shutil
import subprocess
import yaml
from pathlib import Path

# Add parent directories to path to import the version manager
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from evaluators.utils.python_version_manager import PythonVersionManager


def update_polyeval_config(python_executable: str):
    """Update the polyeval_config.yaml with the correct Python executable."""
    config_path = Path(__file__).parent / "project-templates" / "default" / "python" / "polyeval_config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update commands with the detected Python executable
        config['commands']['build'] = f"{python_executable} -m py_compile src/main.py"
        config['commands']['run'] = f"{python_executable} src/main.py"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Updated polyeval_config.yaml to use: {python_executable}")
        return True
        
    except Exception as e:
        print(f"Error updating config: {e}")
        return False


def main():
    """Main setup function."""
    print("\n" + "=" * 70)
    print("PolyHumanEval Python 3.9 Setup Script")
    print("=" * 70)
    
    manager = PythonVersionManager()
    
    # Check Python availability
    available, python_exe, messages = manager.check_python_availability()
    
    print("\n--- Python Version Check ---")
    for msg in messages:
        print(msg)
    
    if available:
        print(f"\n✓ Found suitable Python: {python_exe}")
        version = manager.get_python_version(python_exe)
        print(f"  Version: {version}")
        
        # Update config file
        print("\n--- Updating Configuration ---")
        if update_polyeval_config(python_exe):
            print("✓ Configuration updated successfully!")
        
        # Set environment variable suggestion
        print("\n--- Environment Setup ---")
        print("To make this permanent, add to your shell configuration:")
        print(f"  export POLYHUMANEVAL_PYTHON={python_exe}")
        
        # Test the Python executable
        print("\n--- Testing Python Executable ---")
        try:
            result = subprocess.run(
                [python_exe, "-c", "import sys; print(f'Python {sys.version}')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Python test successful:")
                print(f"  {result.stdout.strip()}")
            else:
                print(f"✗ Python test failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Error testing Python: {e}")
        
    else:
        print("\n✗ No suitable Python version found!")
        print("\nPlease install Python 3.9 and run this script again.")
        
        # Create a shell script for easy installation
        install_script = Path(__file__).parent / "install_python39.sh"
        with open(install_script, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Auto-generated script to install Python 3.9\n\n")
            
            if sys.platform == "darwin":
                f.write("# macOS installation using Homebrew\n")
                f.write("if command -v brew &> /dev/null; then\n")
                f.write("    echo 'Installing Python 3.9 using Homebrew...'\n")
                f.write("    brew install python@3.9\n")
                f.write("else\n")
                f.write("    echo 'Homebrew not found. Please install from https://brew.sh'\n")
                f.write("fi\n")
            elif sys.platform.startswith("linux"):
                f.write("# Linux installation\n")
                f.write("if command -v apt &> /dev/null; then\n")
                f.write("    echo 'Installing Python 3.9 using apt...'\n")
                f.write("    sudo apt update\n")
                f.write("    sudo apt install -y python3.9 python3.9-dev python3.9-venv\n")
                f.write("elif command -v yum &> /dev/null; then\n")
                f.write("    echo 'Installing Python 3.9 using yum...'\n")
                f.write("    sudo yum install -y python39\n")
                f.write("else\n")
                f.write("    echo 'Package manager not recognized. Please install Python 3.9 manually.'\n")
                f.write("fi\n")
        
        install_script.chmod(0o755)
        print(f"\n💡 Quick install script created: {install_script}")
        print(f"   Run: bash {install_script}")
    
    print("\n" + "=" * 70)
    return 0 if available else 1


if __name__ == "__main__":
    sys.exit(main())