import os
from typing import List, Optional

def find_matching_files(
    directory: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
) -> List[str]:
    """
    List all files in a directory (non-recursive) that match optional prefix and/or suffix.

    Args:
        directory (str): Path to the directory.
        prefix (str, optional): File name prefix to match. Defaults to None.
        suffix (str, optional): File name suffix to match. Defaults to None.

    Returns:
        List[str]: Filenames (with full path) that match the pattern.
    """
    files = []
    for fname in os.listdir(directory):
        if prefix and not fname.startswith(prefix):
            continue
        if suffix and not fname.endswith(suffix):
            continue
        files.append(os.path.join(directory, fname))
    return files