import zlib, base64, pickle
import json

def decompress_test_cases(encoded: str):
    """
    Inverse of compress_test_cases().
    NOTE: Uses pickle.loads on trusted data only.
    """
    compressed = base64.b64decode(encoded.encode("utf-8"))
    pickled = zlib.decompress(compressed)
    json_str = pickle.loads(pickled)          # -> the JSON string
    return json.loads(json_str)               # -> original Python object

def compress_test_cases(test_cases):
    """
    Compress a Python object (e.g., test cases) into a base64-encoded string,
    using JSON serialization, pickling, and zlib compression.
    """
    json_str = json.dumps(test_cases)
    pickled = pickle.dumps(json_str)
    compressed = zlib.compress(pickled)
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded