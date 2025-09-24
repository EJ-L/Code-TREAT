# PolyHumanEval Python 3.9 Setup Guide

## Quick Start - Dedicated Virtual Environment (Recommended)

The PolyHumanEval evaluation requires **Python 3.9** for optimal compatibility. The easiest way is to create a dedicated virtual environment that works alongside your main uv-managed project.

### 1. One-Command Setup

Create a dedicated Python 3.9 environment for PolyHumanEval:

```bash
python environment_setup/setup_polyhumaneval_environments.py
```

This script will:
- Find Python 3.9 on your system (from any source: uv, pyenv, system, conda, etc.)
- Create a dedicated virtual environment at `virtual_environments/polyhumaneval/venv/python3.9/`
- Verify the setup works correctly
- Provide installation instructions if Python 3.9 is not found

**That's it!** The code translation evaluator will automatically detect and use this environment.

### 2. How It Works

- **Your main project**: Runs on Python 3.12 via `uv run scripts/run_evaluation.py`
- **PolyHumanEval tests**: Execute in the dedicated Python 3.9 venv automatically
- **No conflicts**: Each environment is completely isolated

## Installing Python 3.9 (If Not Available)

If the setup script can't find Python 3.9, install it using your preferred method:

### Option 1: Using uv (Since you already have it)

```bash
uv python install 3.9
# Then run the setup script again
python environment_setup/setup_polyhumaneval_environments.py
```

### Option 2: Platform-Specific Installation

#### macOS

```bash
# Using Homebrew
brew install python@3.9

# Or using pyenv
brew install pyenv
pyenv install 3.9.18
```

#### Ubuntu/Debian Linux

```bash
# Option 1: Using apt
sudo apt update
sudo apt install python3.9 python3.9-dev python3.9-venv

# Option 2: Using pyenv
curl https://pyenv.run | bash
pyenv install 3.9.18
pyenv global 3.9.18

# Option 3: Using uv
uv python install 3.9
```

#### RHEL/CentOS/Fedora

```bash
# Option 1: Using yum
sudo yum install python39

# Option 2: Using pyenv
curl https://pyenv.run | bash
pyenv install 3.9.18
pyenv global 3.9.18
```

#### Windows

1. Download from [Python.org](https://www.python.org/downloads/release/python-3918/)
2. Run the installer
3. **Important**: Check "Add Python to PATH"

Or using Chocolatey:
```powershell
choco install python39
```

#### Using Conda/Miniconda (All Platforms)

```bash
conda create -n py39 python=3.9
conda activate py39
```

## Your Workflow

After setup, your workflow remains unchanged:

```bash
# Your main project runs normally with uv and Python 3.12
uv run scripts/run_evaluation.py

# PolyHumanEval automatically uses the dedicated Python 3.9 environment
# No additional configuration needed!
```

## Advanced Configuration

You can override the automatic detection if needed:

```bash
# Set environment variable to use a specific Python
export POLYHUMANEVAL_PYTHON=/path/to/python3.9

# Or use system Python 3.9 instead of venv
export POLYHUMANEVAL_PYTHON=python3.9
```

## How the System Works

The evaluation system uses a priority-based Python detection system:

1. **First Priority**: `POLYHUMANEVAL_PYTHON` environment variable (if set)
2. **Second Priority**: Local venv at `virtual_environments/polyhumaneval/venv/python3.9/` (created by setup script)
3. **Third Priority**: System-wide Python 3.9 installations
4. **Fallback**: Python 3.10/3.11 with warnings
5. **Last Resort**: Current Python (with warnings about compatibility issues)

### Detection Process

- **Checks local venv first**: Most reliable and isolated
- **Searches system paths**: Common installation directories
- **Supports all installation methods**: uv, pyenv, conda, homebrew, system packages
- **Version validation**: Ensures it's actually Python 3.9
- **Automatic fallback**: Graceful degradation with clear warnings

### 5. Troubleshooting

#### Python 3.9 installed but not detected?

Check if it's in your PATH:
```bash
which python3.9
# or
where python3.9  # Windows
```

If found, set the environment variable explicitly:
```bash
export POLYHUMANEVAL_PYTHON=/path/to/python3.9
```

#### Permission errors during installation?

- On Linux/macOS, you might need `sudo` for system-wide installation
- Consider using pyenv or conda for user-level installation without sudo

#### Multiple Python versions installed?

The system will automatically prefer Python 3.9. You can override this:
```bash
export POLYHUMANEVAL_PYTHON=python3.9  # or full path
```

### 6. How It Works

The evaluation system uses a `PythonVersionManager` that:

1. **Checks environment variable** `POLYHUMANEVAL_PYTHON` first
2. **Searches for Python 3.9** in common locations:
   - `python3.9`, `python3`, `python`
   - Common installation directories
   - pyenv installations
3. **Validates the version** to ensure it's actually Python 3.9
4. **Falls back gracefully** with warnings if 3.9 is not available
5. **Updates configuration** automatically for polyeval templates

### 7. For Developers

The Python version management is handled by:
- `evaluators/utils/python_version_manager.py` - Version detection and validation
- `evaluators/utils/executor.py` - Configurable Python execution
- `evaluators/code_translation_evaluator.py` - Automatic setup on import

To programmatically check Python availability:

```python
from evaluators.utils.python_version_manager import PythonVersionManager

manager = PythonVersionManager()
available, python_exe, messages = manager.check_python_availability()

if available:
    print(f"Using Python: {python_exe}")
else:
    print("Python 3.9 not found")
    for msg in messages:
        print(msg)
```

### 8. Why Python 3.9?

Python 3.9 is chosen for PolyHumanEval because:
- It's stable and widely available
- Compatible with most test cases
- Balances modern features with broad compatibility
- Matches the version used in the original benchmark design

### 9. Acceptable Alternatives

While Python 3.9 is recommended, these versions are also acceptable:
- Python 3.10
- Python 3.11

Python 3.12+ may have compatibility issues with some test cases.

---

## Summary

1. **Run the setup script** to check your Python configuration
2. **Install Python 3.9** if not available using your preferred method
3. **Set `POLYHUMANEVAL_PYTHON`** environment variable if needed
4. **The system will auto-detect** and configure Python for you

For issues or questions, check the troubleshooting section or run:
```bash
python -c "from evaluators.utils.python_version_manager import print_setup_guide; print_setup_guide()"
```