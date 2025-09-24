from typing import Dict, List, Any, Optional, Tuple, Iterable, Union

def normalize_categories(cat: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    """Coerce categories to a stable, sorted list of strings."""
    if cat is None:
        return []
    if isinstance(cat, str):
        cats = [cat]
    elif isinstance(cat, (list, tuple)):
        cats = [str(c) for c in cat]
    else:
        cats = [str(cat)]
    return sorted(cats)


def normalize_batch(x: Any) -> List[str]:
    """Normalize various model.chat return shapes into List[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        if "choices" in x and isinstance(x["choices"], list):
            texts: List[str] = []
            for c in x["choices"]:
                if isinstance(c, dict):
                    t = c.get("text") or (c.get("message", {}) or {}).get("content")
                    if isinstance(t, str) and t:
                        texts.append(t)
            return texts
        if "text" in x and isinstance(x["text"], str):
            return [x["text"]]
        return []
    if isinstance(x, (list, tuple)):
        return [s for s in x if isinstance(s, str)]
    return []