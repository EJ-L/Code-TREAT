import re
from typing import List, Optional, Tuple, Union

def extract_fenced_code(
    text: str,
    lang: Optional[str] = None,
) -> Tuple[bool, Union[None, str, List[str]]]:
    """
    Extract fenced Markdown code blocks, preferring typed fences first.

    Search order:
      1) Typed code blocks:
         - if lang is provided: ```{lang} [info]\n<code>\n```
         - if lang is None: any typed block ```<token> [info]\n<code>\n```
      2) Untyped code blocks: ```\n<code>\n```

    Returns:
      all_matches=False:
        (True, code)  if exactly one match; else (False, None)
      all_matches=True:
        (True, [code1, ...]) if >=1 matches; else (False, [])
    """
    # Normalize newlines and handle edge cases
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Handle potential nested or malformed fences
    # Only clean up problematic nested patterns, preserve valid fences
    # Look for and fix the specific nested pattern: ```\n```lang\ncode\n```\n```
    nested_pattern = re.compile(r'```\s*\n```([^`\n]*)\n(.*?)\n```\s*\n```', re.DOTALL)
    text = nested_pattern.sub(r'```\1\n\2\n```', text)

    # ------------- build patterns -------------
    # Match backticks length >=3, capture as 'fence' to require same length on close.
    if lang is None:
        # Any typed block: language token followed by optional info string, then newline.
        typed_pat = re.compile(
            r"""(?msx)
            ^[ \t]*(?P<fence>`{3,})[ \t]*      # opening fence
            (?P<lang>[^\s`]+)[^\n]*\n          # language token + optional info
            (?P<code>.*?)                      # code payload
            \n^[ \t]*(?P=fence)[ \t]*$         # closing fence (same length)
            """,
            re.MULTILINE | re.DOTALL | re.VERBOSE,
        )
    else:
        # Specific language (case-insensitive), allow optional info string.
        typed_pat = re.compile(
            rf"""(?msx)
            ^[ \t]*(?P<fence>`{{3,}})[ \t]*    # opening fence
            {re.escape(lang)}\b[^\n]*\n        # language + optional info
            (?P<code>.*?)                      # code payload
            \n^[ \t]*(?P=fence)[ \t]*$         # closing fence (same length)
            """,
            re.MULTILINE | re.DOTALL | re.VERBOSE | re.IGNORECASE,
        )

    # Untyped: nothing but spaces/tabs after opening fence before newline.
    untyped_pat = re.compile(
        r"""(?msx)
        ^[ \t]*(?P<fence>`{3,})[ \t]*\n        # opening fence (no lang/info)
        (?P<code>.*?)                          # code payload
        \n^[ \t]*(?P=fence)[ \t]*$             # closing fence (same length)
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

    # ------------- search typed first -------------
    typed_blocks = [_trim_filename_header(m.group("code")) for m in typed_pat.finditer(text)]
    if typed_blocks:
        return True, typed_blocks

    # ------------- then fallback to untyped -------------
    untyped_blocks = [_trim_filename_header(m.group("code")) for m in untyped_pat.finditer(text)]
    if untyped_blocks:
        return True, untyped_blocks
    
    return False, None


# Optional helper: remove filename headers that can appear in various forms
_FILENAME_HEADERS = [
    # Pattern 1: ":filename.ext" at start of line
    re.compile(
        r"^\s*:\s*[\w\-.]+\.(?:java|py|cpp|cc|cxx|js|ts|c|cs|rb|go|rs|php|swift|kt|scala|dart)\s*\n",
        re.IGNORECASE,
    ),
    # Pattern 2: "filename.ext" at start of line (no colon)
    re.compile(
        r"^[\w\-.]+\.(?:java|py|cpp|cc|cxx|js|ts|c|cs|rb|go|rs|php|swift|kt|scala|dart)\s*\n",
        re.IGNORECASE,
    ),
    # Pattern 3: Common filename patterns in comments
    re.compile(
        r"^//\s*[\w\-.]+\.(?:java|py|cpp|cc|cxx|js|ts|c|cs|rb|go|rs|php|swift|kt|scala|dart)\s*\n",
        re.IGNORECASE,
    ),
    # Pattern 4: Python-style filename comments
    re.compile(
        r"^#\s*[\w\-.]+\.(?:java|py|cpp|cc|cxx|js|ts|c|cs|rb|go|rs|php|swift|kt|scala|dart)\s*\n",
        re.IGNORECASE,
    ),
]

def _trim_filename_header(code: str) -> str:
    """Remove common filename header patterns from the start of code."""
    result = code
    for pattern in _FILENAME_HEADERS:
        result = pattern.sub("", result, count=1)
        # If we removed something, we're done
        if result != code:
            break
    return result
