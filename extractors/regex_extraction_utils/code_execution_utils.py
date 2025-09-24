import re       
def get_input(assertion_statement):
    import re
    pattern = r'(.*)\nexpected_output.*?'
    match = re.search(pattern, assertion_statement, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None 
    
def mask_inputs(line: str) -> str:
    """
    Masks any content after 'inputs =' so that:
      "inputs = (2, 2, [[1, 3], [4, 10]])"  →  "inputs = ??"
    """
    return re.sub(r'^(inputs\s*=\s*).+', r'\1??', line)

def get_expected(assertion_statement):
    import re
    pattern = r'expected_output = (.*)\n'
    match = re.search(pattern, assertion_statement, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    
def get_assertion(assertion_statement):
    import re
    pattern = r'assert\s*(.*)\s*==\s*(.*)'
    match = re.search(pattern, assertion_statement, re.DOTALL)
    if match:
        full_assertion = match.group(0)
        function = match.group(1)
        expected_output = match.group(2)
        masked_assertion = full_assertion.replace(expected_output, "??")
        return function, masked_assertion
    else:
        return None