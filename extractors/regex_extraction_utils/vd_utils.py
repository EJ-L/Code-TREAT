import re
import json
from typing import List, Optional
from evaluators.utils.utils import remove_thinking_tag

def extract_json_from_response(response: str, tolerate: bool = False) -> Optional[dict]:
    """
    Extract JSON object from response string, handling both code fence and raw JSON formats.
    Args:
        response (str): Response string that may contain JSON in code fence or raw format
        tolerate (bool): If True, try to parse non-JSON formatted responses
    Returns:
        dict or None: Extracted JSON object if found and valid, None otherwise
    """
    # Try to extract JSON from code fence first
    code_fence_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_fence_pattern, response, re.DOTALL)
    if match:
        try:
            json_obj = json.loads(match.group(1))
            # Convert numeric values to strings for consistency
            if isinstance(json_obj.get("code1"), (int, str)) and isinstance(json_obj.get("code2"), (int, str)):
                if str(json_obj["code1"]).lower() in ["(1)", "1", "yes"]:
                    code1_result = "YES"
                elif str(json_obj["code1"]).lower() in ["(2)", "2", "no"]:
                    code1_result = "NO"
                else:
                    code1_result = -1

                
                if str(json_obj["code2"]).lower() in ["(1)", "1", "yes"]:
                    code2_result = "YES"
                elif str(json_obj["code2"]).lower() in ["(2)", "2", "no"]:
                    code2_result = "NO"
                else:
                    code2_result = -1
                    
                if code1_result == -1 or code2_result == -1:
                    return None
                
                return {
                    "code1": code1_result,
                    "code2": code2_result
                }
        except json.JSONDecodeError:
            pass

    # Try to extract raw JSON
    try:
        # Find the first occurrence of a JSON-like structure
        # Updated pattern to handle both string and numeric values
        json_pattern = r'\{[^{]*"code1":\s*(?:"[^"]+"|[01])\s*,\s*"code2":\s*(?:"[^"]+"|[01])\s*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            json_obj = json.loads(match.group(0))
            # Convert numeric values to strings for consistency
            if isinstance(json_obj.get("code1"), (int, str)) and isinstance(json_obj.get("code2"), (int, str)):
                if str(json_obj["code1"]).lower() in ["(1)", "1", "yes"]:
                    code1_result = "YES"
                elif str(json_obj["code1"]).lower() in ["(2)", "2", "no"]:
                    code1_result = "NO"
                else:
                    code1_result = -1

                
                if str(json_obj["code2"]).lower() in ["(1)", "1", "yes"]:
                    code2_result = "YES"
                elif str(json_obj["code2"]).lower() in ["(2)", "2", "no"]:
                    code2_result = "NO"
                else:
                    code2_result = -1
                    
                if code1_result == -1 or code2_result == -1:
                    return None
                
                return {
                    "code1": code1_result,
                    "code2": code2_result
                }
    except json.JSONDecodeError:
        pass

    # If tolerance is enabled, try to parse non-JSON format
    if tolerate:
        try:
            # First try to find explicit (1) YES/NO and (2) YES/NO patterns
            code1_patterns = [
                r'\(1\)\s*YES[:\s]',
                r'code1[:\s]*\(1\)\s*YES',
                r'first[^.]*\(1\)\s*YES',
                r'code1[^.]*YES',
                r'for `code1`:[^.]*\(1\)\s*YES'
            ]
            code2_patterns = [
                r'\(2\)\s*NO[:\s]',
                r'code2[:\s]*\(2\)\s*NO',
                r'second[^.]*\(2\)\s*NO',
                r'code2[^.]*NO',
                r'for `code2`:[^.]*\(2\)\s*NO'
            ]
            
            response_lower = response.lower()
            
            # Check for code1 patterns
            code1_result = "NO"
            for pattern in code1_patterns:
                if re.search(pattern.lower(), response_lower):
                    code1_result = "YES"
                    break
                    
            # Check for code2 patterns
            code2_result = "NO"
            for pattern in code2_patterns:
                if re.search(pattern.lower(), response_lower):
                    code2_result = "YES"
                    break
                    
            return {
                "code1": code1_result,
                "code2": code2_result
            }
            
        except Exception:
            pass

    return None

def handle_primevul_pair_response(response: str, tolerate: bool = False) -> List[int]:
    """
    Process a vulnerability detection pair response and convert to binary values.
    Args:
        response (str): The response string containing vulnerability assessment for two code snippets
        tolerate (bool): If True, try to parse non-JSON formatted responses
    Returns:
        List[int]: List of two integers [code1_result, code2_result] where:
                  1 = vulnerability detected
                  0 = no vulnerability
                  -1 = invalid/None response
    """
    if response is None:
        return [-1, -1]

    json_data = extract_json_from_response(response, tolerate)
    if json_data is None:
        return [-1, -1]

    results = []
    for key in ['code1', 'code2']:
        value = json_data.get(key, '').lower()
        if 'yes' in value:
            results.append(1)
        elif 'no' in value:
            results.append(0)
        else:
            results.append(-1)

    return results

def handle_primevul_response(response: str) -> int:
    """
    Convert a vulnerability detection response to a binary value with strict pattern matching.
    Args:
        response (str): The response string indicating vulnerability detection result
    Returns:
        int: 1 if vulnerability detected
             0 if no vulnerability
             -1 if response is None
             -2 if pattern is not recognized (needs manual review)
    """
    # Handle None response
    if response is None:
        return -1
    response = remove_thinking_tag(response)
    # First strip whitespace but preserve case for exact matching
    response_orig = response.strip()
    
    # Handle simple yes/no responses case-insensitively
    response_upper = response_orig.upper()
    if response_upper in ["YES", "YES.", "(1)"]:
        return 1
    if response_upper in ["NO", "NO.", "(2)"]:
        return 0
        
    # Clean and normalize the response for pattern matching
    response = response_orig.lower()
    
    # Handle numeric responses
    if response.isdigit():
        if response == "1":
            return 1
        elif response == "0" or response == "2":
            return 0
        else:
            return -2
            
    # Explicit YES patterns - these are the only patterns we accept for positive detection
    yes_patterns = [
        "(1) yes: a security vulnerability detected",
        "(1) yes",
        "yes: a security vulnerability detected",
        "1. yes"
    ]
    
    # Explicit NO patterns - these are the only patterns we accept for negative detection
    no_patterns = [
        "(2) no: no security vulnerability detected",
        "(2) no",
        "(2)",  # Add this pattern explicitly
        "no: no security vulnerability detected",
        "no security vulnerability",
        "(2) no: no security vulnerability."
    ]
    
    # Check for exact matches in yes patterns
    if any(pattern in response for pattern in yes_patterns):
        # Double check there's no contradiction with no patterns
        if not any(pattern in response for pattern in no_patterns):
            return 1
            
    # Check for exact matches in no patterns
    if any(pattern in response for pattern in no_patterns):
        return 0
    if response.split('\n')[-1].lower() == "no":
        return 0
    if response.split("\n")[-1].lower() == "yes":
        return 1
    # If we get here, the pattern wasn't recognized
    return -2

def handle_response(response: str, dataset: str, tolerate: bool = False):
    """
    Process and parse LLM responses for vulnerability detection tasks based on the dataset type.
    
    This function serves as a router to direct the response handling to the appropriate specialized
    function based on the dataset type. It supports handling both single vulnerability detection
    (primevul) and paired vulnerability comparison (primevul_pair) tasks.
    
    Args:
        response (str): The LLM's response string to be processed. Can be None, in which case
                       appropriate default values will be returned (-1 for primevul, [-1, -1] for primevul_pair)
        dataset (str): The type of dataset being processed. Must be one of:
                      - "primevul": Single vulnerability detection
                      - "primevul_pair": Paired vulnerability comparison
        tolerate (bool, optional): If True, enables more lenient parsing for non-standard response formats.
                                 Defaults to False.
    
    Returns:
        Union[int, List[int]]: The parsed result, which can be:
            - For primevul dataset:
                int: where 1=vulnerable, 0=not vulnerable, -1=invalid/None response, -2=unrecognized pattern
            - For primevul_pair dataset:
                List[int]: [code1_result, code2_result] where for each:
                          1=vulnerable, 0=not vulnerable, -1=invalid/None response
    
    Raises:
        ValueError: If the dataset parameter is not one of the supported types
    
    Example:
        >>> handle_response("(1) yes", "primevul")
        1
        >>> handle_response(None, "primevul_pair")
        [-1, -1]
        >>> handle_response('{"code1": "YES", "code2": "NO"}', "primevul_pair", tolerate=True)
        [1, 0]
    """
    if dataset == 'primevul':
        return handle_primevul_response(response)
    elif dataset == 'primevul_pair':
        return handle_primevul_pair_response(response, tolerate)
    else:
        raise ValueError(f"Unknown dataset: {dataset}, you may need to self-customize the response handling function!")
