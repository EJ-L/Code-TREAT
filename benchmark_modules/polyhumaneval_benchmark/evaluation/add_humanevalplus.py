# Read a TestDSL file
def read_testdsl(file_path):
    problems = {}
    with open(file_path, 'r') as file:
        cur_key = None
        code_bracket_count = 0
        template_nse_bracket_count = 0
        tests_bracket_count = 0
        for line in file:
            if line.startswith("problem HumanEval"):
                if cur_key is not None:
                    problems[cur_key]['code'].append("    }\n")
                    problems[cur_key]['template_nse'].append("        }\n")
                    problems[cur_key]['test_cases'].append("    }\n")
                
                cur_key = line.split(" {")[0]
                problems[cur_key] = {}
                problems[cur_key]['full_problem'] = []
                problems[cur_key]['test_cases'] = []
                problems[cur_key]['code'] = []
                problems[cur_key]['template_nse'] = []
            problems[cur_key]['full_problem'].append(line)
            if line.find("code {") != -1:
                code_bracket_count += 1
            if line.find("template nse {") != -1:
                template_nse_bracket_count += 1
            if line.find("tests {") != -1: 
                tests_bracket_count += 1
                
            if line.find("}") != -1:
                if code_bracket_count != 0:
                    code_bracket_count -= 1
                if template_nse_bracket_count != 0:
                    template_nse_bracket_count -= 1
                if tests_bracket_count != 0:
                    tests_bracket_count -= 1
                
            if code_bracket_count != 0:    
                problems[cur_key]['code'].append(line)
            if template_nse_bracket_count != 0:
                problems[cur_key]['template_nse'].append(line)
            if tests_bracket_count != 0:
                problems[cur_key]['test_cases'].append(line)

    return problems

def extract_check_function(code):
    lines = code.split("\n")
    inputs = None
    outputs = None
    for idx, line in enumerate(lines):
        if "def check" in line:
            function_impl = lines[idx+1:]
            break
    has_results = True if "results" in "".join(function_impl) else False
    for line in function_impl:
        if "inputs" in line:
            # print(inputs)
            inputs = literal_eval(line.split("inputs = ")[1])
            if not has_results:
                outputs = [''] * len(inputs)
        if "results" in line and has_results:
            outputs = literal_eval(line.split("results = ")[1])
        if inputs is not None and outputs is not None:
            input_output_pairs = list(zip(inputs, outputs))
            return input_output_pairs, has_results
    return code, has_results

def convert_value_to_type(value, type_hint):
    """Convert a value to match the function's type signature"""
    # Handle nullable types
    if type_hint.endswith('?'):
        if value is None:
            return None
        return convert_value_to_type(value, type_hint[:-1])
        
    # Handle nested lists
    if type_hint.startswith("list<"):
        if not isinstance(value, list):
            return value
        inner_type = type_hint[5:-1]
        return [convert_value_to_type(x, inner_type) for x in value]
        
    # Handle basic types
    if type_hint == "int":
        try:
            return int(float(value)) if value is not None else None
        except (ValueError, TypeError):
            return value
    elif type_hint == "double":
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return value
    elif type_hint == "bool":
        return bool(value) if value is not None else None
    elif type_hint == "string":
        return str(value) if value is not None else None
        
    return value

def extract_type_hints(code_lines):
    """Extract parameter and return type hints from function signature"""
    if isinstance(code_lines, str):
        code_lines = code_lines.split('\n')
    
    for line in code_lines:
        if "func" in line and "->" in line:
            print("Found function signature:", line)  # Debug
            # Parse function signature
            params_part = line[line.find("(")+1:line.find(")")].strip()
            print("Parameters part:", params_part)  # Debug
            
            # Split parameters more carefully
            params = []
            if params_part:  # Only split if there are parameters
                params = [p.split(":") for p in params_part.split(",") if p.strip()]
            print("Parsed parameters:", params)  # Debug
            
            # Safer dictionary comprehension
            param_types = {}
            for p in params:
                if len(p) == 2:  # Only add if we have both name and type
                    param_types[p[0].strip()] = p[1].strip()
            
            return_type = line.split("->")[1].strip()
            print("Return type:", return_type)  # Debug
            return param_types, return_type
    
    print("No function signature found in:", code_lines)  # Debug
    return {}, None

def format_test_case(input_value, output_value, param_types):
    """Format test case with correct types"""
    # Convert input values according to parameter types
    converted_inputs = []
    for value, type_hint in zip(input_value, param_types.values()):
        converted_value = convert_value_to_type(value, type_hint)
        converted_inputs.append(converted_value)
    
    # Format the test case line
    # Handle string output_value - add quotes if it's a string
    if isinstance(output_value, str):
        output_str = f'"{output_value}"'
    else:
        output_str = str(output_value)
    
    data_string = f"            {converted_inputs} -> {output_str}"
    data_string = data_string.replace("[", "(", 1)
    if "]" in data_string:
        last_bracket_pos = data_string.split(" -> ")[0].rindex("]")
        data_string = data_string[:last_bracket_pos] + ")" + data_string[last_bracket_pos + 1:]
    
    # Convert to proper format
    data_string = data_string.replace("'", '"')  # Convert single quotes to double quotes
    data_string = data_string.replace("True", "true")
    data_string = data_string.replace("False", "false")
    data_string = data_string.replace("None", "null")
    return data_string + "\n"

def validate_type(value, type_hint):
    """Check if value matches the expected type"""
    try:
        # Handle nullable types (type?)
        if type_hint.endswith('?'):
            return value is None or validate_type(value, type_hint[:-1])
            
        # Handle nested lists
        if type_hint.startswith("list<"):
            if not isinstance(value, list):
                return False
            # Empty list is valid for any list type
            if not value:
                return True
            # Extract inner type from list<type>
            inner_type = type_hint[5:-1]  # Remove 'list<' and '>'
            # Validate each element
            return all(validate_type(x, inner_type) for x in value)
            
        # Basic types
        if type_hint == "int":
            if isinstance(value, (int, float)):
                try:
                    int_val = int(float(value))
                    # Check Java int range
                    return -2147483648 <= int_val <= 2147483647
                except (OverflowError, ValueError):
                    return False
            return False
        elif type_hint == "double":
            try:
                float(value)
                return isinstance(value, (int, float))
            except (OverflowError, ValueError):
                return False
        elif type_hint == "bool":
            return isinstance(value, bool)
        elif type_hint == "string":
            return isinstance(value, str)
        elif type_hint == "null":
            return value is None
            
        return True  # For unknown types, accept all values
    except Exception as e:
        print(f"Type validation error for {value} with type hint {type_hint}: {e}")
        return False

def filter_valid_test_cases(input_output_pairs, param_types, return_type):
    """Filter test cases that match the function signature types"""
    valid_pairs = []
    # print(len(input_output_pairs))
    for inp, out in input_output_pairs:
        # print(1)
        # print(inp, out)
        # # Check if input types match
        # # if len(inp) != len(param_types):
        # #     print(len(inp), len(param_types))
        # #     continue
        # # print(inp, out)
        # # print(param_types.values())
        # # Validate each input parameter
        # print(2)
        # print(list(zip(inp, param_types.values())))
        # print("input: ", zip(inp, param_types.values()))
        # print("output:", (out, return_type))
        inputs_valid = all(
            validate_type(value, type_hint) 
            for value, type_hint in zip(inp, param_types.values())
        )
        # print(inputs_valid)
        # Validate output if return type is specified
        if return_type and not validate_type(out, return_type):
            continue
            
        if not inputs_valid:
            continue

        # Add valid test case
        valid_pairs.append((inp, out))

    return valid_pairs

# Main execution starts here
from datasets import load_dataset
from ast import literal_eval

# Example usage
file_path = '/Users/ericjohnli/Downloads/LiveCodeBench/test/polyhumaneval_module/evaluation/data/poly_humaneval.testdsl'
testdsl_content = read_testdsl(file_path)

test_data = load_dataset("evalplus/humanevalplus")['test']
def escape_string(s: str):
    new_s = []
    for c in s:
        if c == "\\":
            new_s.append("\\\\")
        elif c == "\n":
            new_s.append("\\n")
        elif c == "\t":
            new_s.append("\\t")
        elif c == "\r":
            new_s.append("\\r")
        else:
            new_s.append(c)
    return "".join(new_s)
import unicodedata
import unicodedata

def contains_emoji(s: str) -> bool:
    for s_ in s:
        print(s_)
        for char in s_:
            print(char)
            if unicodedata.category(char) in {"So", "Sk"}:
                return True
    return False

def extract_emojis(s: str) -> str:
    # Normalize the string to handle surrogate pairs
    s = s.encode("utf-16", "surrogatepass").decode("utf-16")
    
    # Extract and return all emoji characters
    return "".join(char for char in s if unicodedata.category(char) in {"So", "Sk"})
import typing
def read_solution():
    import json
    with open("/Users/ericjohnli/Downloads/LiveCodeBench/test/polyhumaneval_module/evaluation/data/poly_humaneval_sol.json", "r") as file:
        json_data = json.load(file)
    return json_data

from math import floor
import re
# Create a namespace dictionary


gold_solution = read_solution()
# Iterate through each row of the test data
for i in range(0, 164):
    # 15 missed out
    # if i not in [14, 15, 21, 25, 27, 28, 28, 30, 30, 32, 32, 33, 38, 45, 45, 47, 49, 50, 72, 74, 76, 89, 101, 103, 112, 118, 122, 123, 125, 132, 137, 141, 144, 145, 155, 160, 161]:
    # if i != 112:
    if i != 140:
        continue
    
    row = test_data[i]
    input_output_pairs, has_results = extract_check_function(row['test'])
    print(f"HumanEval/{i}")
    # Add debug print here
    # print("Input-output pairs before filtering:", input_output_pairs)  # DEBUG
    
    # Find the matching problem key
    current_key = None
    for key in testdsl_content.keys():
        if row['task_id'] in key:
            current_key = key
            break
    
    if not current_key:
        continue

    # Extract type hints for this specific problem
    param_types, return_type = extract_type_hints(testdsl_content[current_key]['code'])
    
    # Execute function first to get results
    ref_func = gold_solution['python'][f'HumanEval/{i}']
    # namespace = {}
    namespace = {k: getattr(typing, k) for k in typing.__all__}
    namespace['floor'] = floor
    namespace['re'] = re
    exec(ref_func, namespace)
    
    # Get the function by name and call it
    func_name = row['entry_point']
    new_input_output_pairs = []
    # Get results for all test cases first
    from tqdm import tqdm
    for idx, (inp, output) in enumerate(tqdm(input_output_pairs)):
        next_element = False
        print(inp)
        # for inp_ in inp[0]:
        #     def digits_sum(n: int) -> int:
        #         n_str = str(n)
        #         overflow = False
        #         if n >= 0:
        #             if not - 2 ** 31 < sum(int(d) for d in n_str) < 2 ** 31 - 1:
        #                 overflow = True
        #                 print("overflow")
        #             return sum(int(d) for d in n_str), overflow
        #         else:
        #             if not - 2 ** 31 < int(n_str[:2]) * 2 < 2 ** 31 - 1:
        #                 overflow = True
        #                 print("overflow")
        #             if not - 2 ** 31 < (int(n_str[:2]) * 2 + digits_sum(abs(n))[0]) < 2 ** 31 - 1:
        #                 overflow = True
        #                 print("overflow")
        #             return int(n_str[:2]) * 2 + digits_sum(abs(n))[0], overflow or digits_sum(abs(n))[1]
        #     _, overflow = digits_sum(inp_)
        #     if overflow:
        #         print("Overflow:", overflow)
        #         next_element = True
        #         break
        # for 144
        # xs = inp[0].split("/")
        # xs = [int(s) for s in xs]
        # ns = inp[1].split("/")
        # ns = [int(s) for s in ns]
        # if xs[0] * ns[0] > 2 ** 31 - 1 or xs[0] * ns[0] < - 2**31:
        #     next_element = True
        # if xs[1] * ns[1] > 2 ** 31 - 1 or xs[1] * ns[1] < - 2**31:
        #     next_element = True
        # if next_element:
        #     continue
        try:
            # print(inp)
            # if int(inp[0]) > 2 ** 31 - 1:
            #     continue
            if i == 12 or i == 28 or i == 161:
                for inp_ in inp:                    
                    print(inp_)
                    if contains_emoji(inp_):
                        next_element = True
                        print(next_element)
                        break
            if next_element:
                print("next")
                continue
            result = namespace[func_name](*inp)
            print(result)
            if isinstance(result, str):
                # Attempt to convert the string to an int, then to a float if the first conversion fails
                print("string")
                try:
                    converted_result = int(result)
                except:
                    try:
                        converted_result = float(result)
                        print("float")
                    except:
                        # If both conversions fail, escape the string
                        converted_result = escape_string(result)
                        print("string")
                        if len(converted_result) > 2 ** 31 - 1:
                            continue
                # Append the input-output pair to the list
                new_input_output_pairs.append((inp, converted_result))
            elif isinstance(result, list):
                print("list")
                new_result = []
                for result_ in result:
                    try:
                        result_ = float(result_)
                    except:
                        try:
                            result_ = int(result_)
                        except:
                            print("string")
                            # result_ = escape_string(result_)
                    new_result.append(result_)
                new_input_output_pairs.append((inp, new_result))
            else:
                new_input_output_pairs.append((inp, result))
                    
        except Exception as e:
            print(f"Error executing test case {idx}: {e}")
            continue
    print("after manipulation:")
    # print(new_input_output_pairs)
    input_output_pairs = new_input_output_pairs
    # Add debug print here
    # print("Input-output pairs after execution:", input_output_pairs)  # DEBUG
    # if i in [8, 20, 107, 112, 136, 148, 155]:
    #     input_output_pairs = [(inp, list(out)) for inp, out in input_output_pairs]
    # if i == 112:
    #     for idx, (inp, out) in enumerate(input_output_pairs):
    #         output1, output2 = out
    #         if output2 == True:
    #             output2 = "yes"
    #         else:
    #             output2 = "no"
    #         input_output_pairs[idx] = [inp, [output1, output2]]
        # print(input_output_pairs[0])
    # Now filter test cases based on types
    input_output_pairs = filter_valid_test_cases(
        input_output_pairs,
        param_types, 
        return_type
    )
    if not input_output_pairs:  # Skip if no valid test cases
        continue

    # Format the valid test cases
    lines = []
    for inp, output in input_output_pairs:
        formatted_line = format_test_case(
            inp,
            output,
            param_types
        )
        lines.append(formatted_line)

    # Update the test cases for this problem
    testdsl_content[current_key]['template_nse'] = (
        "    tests {\n"
        "        template nse {\n"
        + "".join(lines)
        + "        }\n"
        "    }\n"
    )

# Sort and write output
keys = list(testdsl_content.keys())
keys.sort(key=lambda x: int(x.split("/")[1]) if len(x.split("/")) > 1 else x)

with open('poly_humanevalv3.testdsl', 'a') as file:
    for key in keys:
        file.write(key + " {\n")
        file.write("".join(testdsl_content[key]['code']))
        if isinstance(testdsl_content[key]['template_nse'], list):
            file.write("".join(testdsl_content[key]['template_nse']))
        else:
            file.write(testdsl_content[key]['template_nse'])
        file.write('}\n')
