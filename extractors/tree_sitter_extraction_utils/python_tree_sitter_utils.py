from tree_sitter_language_pack import get_parser, get_language
from typing import List, Dict
import ast

PYTHON_PARSER, PYTHON_LANG = get_parser('python'), get_language('python')
PYTHON_BASE_IMPORTS = "from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat\nfrom copy import deepcopy\nfrom string import ascii_lowercase\nfrom math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt\nfrom collections import defaultdict, deque, Counter\nfrom bisect import bisect, bisect_left, bisect_right, insort\nfrom heapq import heappush, heappop, heapify, merge\nfrom functools import reduce, cache, lru_cache\nfrom random import randrange, shuffle\nfrom operator import itemgetter, sub\nfrom re import search as re_search  # Assuming 're' refers to a regex search\nfrom os.path import commonprefix\nfrom typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator\nimport os\nimport copy\nimport string\nimport math\nimport collections\nimport bisect\nimport heapq\nimport functools\nimport random\nimport itertools\nimport operator\nimport re\nimport numpy as np\nimport pandas as pd\nfrom math import log, prod  # 'log' and 'prod' are functions in the math module\nfrom collections import deque, defaultdict, Counter, OrderedDict\nfrom itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle\nfrom functools import lru_cache, reduce, partial\n# from sortedcontainers import SortedList, SortedDict, SortedSet\n# import sortedcontainers\nfrom operator import iand\nimport sys\nimport re\nsys.set_int_max_str_digits(1000000)\nsys.setrecursionlimit(10000)\n"

def get_node_text(src, node):
    return src[node.start_byte:node.end_byte].decode('utf-8')

def get_python_imports(code: str) -> List[str]:
    query = """    
        (import_statement) @import_statement
        (import_from_statement) @import_from_statement
    """
    import_stmts = ""
    import_lines = []
    src = bytes(code, 'utf-8')
    tree = PYTHON_PARSER.parse(src)
    query = PYTHON_LANG.query(query)
    target = query.matches(tree.root_node)
    if not target:
        return import_stmts, import_lines
    for _match in target:
        _id, _match_dict = _match
        if _id == 0:
            import_lines.append(get_node_text(src, _match_dict['import_statement'][0]))
        else:
            import_from_statement = get_node_text(src, _match_dict['import_from_statement'][0])
            if 'from typing import staticmethod' in import_from_statement:
                continue
            import_lines.append(import_from_statement.strip())
    import_stmts = '\n'.join(import_lines)
    return import_stmts, import_lines


####### CODE TRANSLATION HELPER ##########
# borrowed from LCB
def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code

recursion_imports = "import sys\nimport re\nsys.set_int_max_str_digits(1000000)\nsys.setrecursionlimit(10000)\n"
def get_node_text(src, node):
    return src[node.start_byte:node.end_byte].decode('utf-8')

### This is specifically design for PolyHumanEval, use this carefully
def extract_python_function_under_class(code: str) -> str:
    """
    Extracts all function definitions under a class definition.
    """
    import_query = """
    (import_statement) @import_statement
    (import_from_statement) @import_from_statement
    """
    query = """
        (class_definition
            name: (identifier) @class.name
            body: (block
                (decorated_definition
                    definition: (function_definition
                        name: (identifier)
                        parameters: (parameters)
                        body: (block)
                    )
                ) @decorator_func
            )
        )
    """

    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    extract_query = PYTHON_LANG.query(query)
    target = extract_query.matches(tree.root_node)
    funcs = []
    if not target:
        return code
    import_query = PYTHON_LANG.query(import_query)
    import_target = import_query.matches(tree.root_node)
    imports = []
    for _match in import_target:
        _id, _match_dict = _match
        if _id == 0:
            imports.append(get_node_text(src, _match_dict['import_statement'][0]))
        else:
            imports.append(get_node_text(src, _match_dict['import_from_statement'][0]))
    import_stmts = '\n'.join(imports)
    for _match in target:
        _id, _match_dict = _match
        if _id == 0: # this is the case of having decorator
            class_name = get_node_text(src, _match_dict['class.name'][0])
            method_code = get_node_text(src, _match_dict['decorator_func'][0])
            method_code_lines = method_code.splitlines()
            decorator = method_code_lines[0].strip()
            if decorator.startswith("@staticmethod"):
                method_code_lines = method_code_lines[1:]
                # dedent each line by 4 spaces if present
                dedented_lines = [
                    line[4:] if line.startswith(" " * 4) else line
                    for line in method_code_lines
                ]

                temp = "\n".join(dedented_lines)
                temp = temp.replace(f"{class_name}.", "")
                funcs.append(temp)
                # code = code.replace("(self, ", "(") # no self in static_method
            else:
                # Dedent lines normally (no decorator line to skip)
                dedented_lines = [
                    line[4:] if line.startswith(" " * 4) else line
                    for line in method_code_lines
                ]
                code = "\n".join(dedented_lines)
                # Remove the 'self,' argument from the function signature
                code = code.replace("(self, ", "(")
                code = code.replace("(self)", "()")
                # Replace class references with plain function references
                code = code.replace(f"{class_name}.", "")
        if _id == 1: # this is the case of typical func
            print(code)
            raise ValueError("THIS SHOULDN'T HAPPEN")
    # new_code = ""
    return recursion_imports + '\n' + import_stmts + '\n' + '\n'.join(funcs)

def get_last_function(matches):
    """
    Given a list of query matches like:
    [(0, {"func": [Node], "name": [Node]}), ...]
    return the dict of the last occurring function (by start_byte).
    """
    if not matches:
        return None

    # Flatten into list of (func_node, match_dict)
    funcs = []
    for _, captures in matches:
        if captures.get("name", []) == "main":
            continue
        for func_node in captures.get("func", []):
            funcs.append((func_node, captures))

    if not funcs:
        return None

    # Pick the function with the max start_byte (latest in file)
    last_func_node, match_dict = max(funcs, key=lambda x: x[0].end_byte)
    return {"func": last_func_node, **match_dict}

def align_func_name(code: str, func_name: str) -> str:
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (identifier) @name
        ) @func
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    last_func = get_last_function(matches)
    # if not last_func:
    #     return code
    last_func_name_start = last_func['name'][0].start_byte
    last_func_name_end = last_func['name'][0].end_byte
    renamed_full_code = src[:last_func_name_start] + bytes(func_name, "utf-8") + src[last_func_name_end:]
    renamed_full_code = align_callee(renamed_full_code.decode("utf-8"), get_node_text(src, last_func['name'][0]), func_name)
    
    return renamed_full_code.decode("utf-8")

def align_callee(code: str, callee_original_name: str, aligned_func_name: str) -> str:
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (call
            function: (_) @callee_name
        )
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    for match in matches:
        _, match = match
        if get_node_text(src, match['callee_name'][0]) == callee_original_name:
            src = src[:match['callee_name'][0].start_byte] + bytes(aligned_func_name, "utf-8") + src[match['callee_name'][0].end_byte:]
    return src

#################### V2 ##################
def get_last_n_functions(matches, n=1, src_bytes=None):
    """
    Return the last n functions (by end_byte) from Tree-sitter matches.
    Skips function named 'main'.
    """
    funcs = []
    for _, captures in matches:
        name_nodes = captures.get("name", [])
        if name_nodes and src_bytes:
            name_text = src_bytes[name_nodes[0].start_byte:name_nodes[0].end_byte].decode()
            if name_text == "main":
                continue
        for func_node in captures.get("func", []):
            funcs.append((func_node, captures))

    if not funcs:
        return []

    funcs.sort(key=lambda x: x[0].end_byte)
    last_funcs = funcs[-n:]
    return [{"func": fn, **caps} for fn, caps in last_funcs]

def align_func_name(code: str, func_names: List[str]) -> str:
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (identifier) @name
        ) @func
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    last_func = get_last_n_functions(matches, len(func_names))
    # if not last_func:
    #     return code
    original_names = []
    renamed_full_code = src
    for func_name, func in zip(func_names[::-1], last_func[::-1]):
        last_func_name_start = func['name'][0].start_byte
        last_func_name_end = func['name'][0].end_byte
        renamed_full_code = renamed_full_code[:last_func_name_start] + bytes(func_name, "utf-8") + renamed_full_code[last_func_name_end:]
        original_name = get_node_text(src, func['name'][0])
        renamed_full_code = align_callee(renamed_full_code.decode("utf-8"), original_name, func_name)
        original_names.append(original_name)
    renamed_full_code = renamed_full_code.decode("utf-8")
    for original_name, new_name in zip(original_names[::-1], func_names):
        renamed_full_code = renamed_full_code.replace(original_name, new_name)
    return renamed_full_code

def python_polyhumaneval_formatter(code: str, ref_name: str) -> str:
    code = extract_python_function_under_class(code)
    code = align_func_name(code, ref_name)
    return code

def align_callee(code: str, callee_original_name: str, aligned_func_name: str) -> str:
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (call
            function: (_) @callee_name
        )
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    for match in matches[::-1]:
        _, match = match
        if get_node_text(src, match['callee_name'][0]) == callee_original_name:
            src = src[:match['callee_name'][0].start_byte] + bytes(aligned_func_name, "utf-8") + src[match['callee_name'][0].end_byte:]
    return src

def python_polyhumaneval_formatter(code: str, ref_name: str) -> str:
    code = extract_python_function_under_class(code)
    code = align_func_name(code, ref_name)
    return code


########## Code Generation #########
def get_func_position_info(code: str) -> list:
    """Extract function definitions with their positions, preferring decorated versions."""
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    
    funcs = []
    decorated_func_ranges = set()
    
    # First pass: collect all decorated function definitions
    decorated_query = """
        (decorated_definition 
            definition: (function_definition) @inner_func
        ) @decorated_func
    """
    extraction_query = PYTHON_LANG.query(decorated_query)
    matches = extraction_query.matches(tree.root_node)
    
    for match in matches:
        _, _match = match
        decorated_node = _match['decorated_func'][0]
        inner_func_node = _match['inner_func'][0]
        
        # Add the full decorated definition (includes decorator + function)
        funcs.append([decorated_node.start_byte, decorated_node.end_byte, get_node_text(src, decorated_node)])
        
        # Remember the inner function's range to avoid duplicates
        decorated_func_ranges.add((inner_func_node.start_byte, inner_func_node.end_byte))
    
    # Second pass: collect standalone functions (not decorated)
    standalone_query = """
        (function_definition) @func
    """
    extraction_query = PYTHON_LANG.query(standalone_query)
    matches = extraction_query.matches(tree.root_node)
    
    for match in matches:
        _, _match = match
        func_node = _match['func'][0]
        
        # Only add if this function is not already captured as part of a decorated definition
        if (func_node.start_byte, func_node.end_byte) not in decorated_func_ranges:
            funcs.append([func_node.start_byte, func_node.end_byte, get_node_text(src, func_node)])
    
    return funcs

def get_class_position_info(code: str) -> list:
    """Extract class definitions with their positions, preferring decorated versions."""
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    
    classes = []
    decorated_class_ranges = set()
    
    # First pass: collect all decorated class definitions
    decorated_query = """
        (decorated_definition 
            definition: (class_definition) @inner_class
        ) @decorated_class
    """
    extraction_query = PYTHON_LANG.query(decorated_query)
    matches = extraction_query.matches(tree.root_node)
    
    for match in matches:
        _, _match = match
        decorated_node = _match['decorated_class'][0]
        inner_class_node = _match['inner_class'][0]
        
        # Add the full decorated definition (includes decorator + class)
        classes.append([decorated_node.start_byte, decorated_node.end_byte, get_node_text(src, decorated_node)])
        
        # Remember the inner class's range to avoid duplicates
        decorated_class_ranges.add((inner_class_node.start_byte, inner_class_node.end_byte))
    
    # Second pass: collect standalone classes (not decorated)
    standalone_query = """
        (class_definition) @class
    """
    extraction_query = PYTHON_LANG.query(standalone_query)
    matches = extraction_query.matches(tree.root_node)
    
    for match in matches:
        _, _match = match
        class_node = _match['class'][0]
        
        # Only add if this class is not already captured as part of a decorated definition
        if (class_node.start_byte, class_node.end_byte) not in decorated_class_ranges:
            classes.append([class_node.start_byte, class_node.end_byte, get_node_text(src, class_node)])
    
    return classes

def get_global_variables(code: str) -> list:
    """Extract global variable assignments at module level."""
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = """
        (assignment) @assignment
        (expression_statement
            (assignment) @assignment
        )
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    
    variables = []
    module_level_assignments = []
    
    # Get all direct children of the module
    for child in tree.root_node.children:
        if child.type == "assignment":
            module_level_assignments.append(child)
        elif child.type == "expression_statement":
            for grandchild in child.children:
                if grandchild.type == "assignment":
                    module_level_assignments.append(grandchild)
    
    # Convert to our format
    for assignment_node in module_level_assignments:
        variables.append([assignment_node.start_byte, assignment_node.end_byte, get_node_text(src, assignment_node)])
    
    return variables

def is_nested_within(inner_item: list, outer_items: list) -> bool:
    """
    Check if an item (function/class) is nested within any other item.
    Returns True only if the item is truly nested (not a direct child at module level).
    """
    inner_start, inner_end, _ = inner_item
    
    for outer_item in outer_items:
        outer_start, outer_end, _ = outer_item
        # Skip self-comparison
        if inner_start == outer_start and inner_end == outer_end:
            continue
        # Check if inner item is completely within outer item
        if outer_start < inner_start and inner_end < outer_end:
            return True
    return False

def is_module_level_item(item: list, code: str) -> bool:
    """
    Check if an item (function/class) is defined at module level.
    This uses Tree-sitter to check the actual nesting level.
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    item_start, item_end, _ = item
    
    # Find the node that corresponds to this item
    def find_node_at_position(node, start_byte, end_byte):
        if node.start_byte == start_byte and node.end_byte == end_byte:
            return node
        for child in node.children:
            result = find_node_at_position(child, start_byte, end_byte)
            if result:
                return result
        return None
    
    target_node = find_node_at_position(tree.root_node, item_start, item_end)
    if not target_node:
        return False
    
    # Check if the parent is the module (root node)
    parent = target_node.parent
    if not parent:
        return True
    
    # For decorated definitions, check the parent of the decorated_definition
    if parent.type == "decorated_definition":
        parent = parent.parent
    
    # If parent is the module root, it's module-level
    return parent == tree.root_node

def filter_nested_items(functions: list, classes: list, code: str) -> tuple:
    """
    Remove all nested functions and classes to keep only top-level definitions.
    This handles:
    - Functions within functions
    - Classes within functions  
    - Functions within classes
    - Classes within classes
    """
    # Filter functions: keep only module-level functions
    top_level_functions = []
    for func in functions:
        if is_module_level_item(func, code):
            top_level_functions.append(func)
    
    # Filter classes: keep only module-level classes
    top_level_classes = []
    for cls in classes:
        if is_module_level_item(cls, code):
            top_level_classes.append(cls)
    
    return top_level_functions, top_level_classes

def early_variable_declarations(code: str) -> str:
    """
    Move variable declarations to the top of functions.
    Fixed version of the original function.
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = """
        (function_definition
            name: (identifier) @func_name
            body: (block) @func_body
        ) @func
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    
    if not matches:
        return code
    
    # Assume single function definition in code snippet
    _, match = matches[0]
    func_body_node = match['func_body'][0]
    body_text = get_node_text(src, func_body_node)
    body_lines = body_text.splitlines()
    
    var_decls = []
    other_lines = []
    
    for line in body_lines:
        stripped_line = line.strip()
        # Look for variable assignments that might be declarations
        if ('=' in stripped_line and 
            not stripped_line.startswith(('if ', 'for ', 'while ', 'def ', 'class ', 'import ', 'from '))):
            var_decls.append(line)
        else:
            other_lines.append(line)
    
    # Reconstruct function body with var declarations at the top
    new_body = '\n'.join(var_decls + other_lines)
    
    # Replace old body with new body in the original code
    new_code = (
        src[:func_body_node.start_byte].decode('utf-8') +
        new_body +
        src[func_body_node.end_byte:].decode('utf-8')
    )
    
    return new_code

def analyze_variable_usage(code: str) -> dict:
    """
    Analyze where variables are used in the code.
    Returns a dict with variable names and their usage patterns.
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    
    # Get all function and class ranges
    functions = get_func_position_info(code)
    classes = get_class_position_info(code)
    
    # Combine function and class ranges
    inside_ranges = []
    for func_start, func_end, _ in functions:
        inside_ranges.append((func_start, func_end))
    for class_start, class_end, _ in classes:
        inside_ranges.append((class_start, class_end))
    
    # Find all identifier usages
    query = """
        (identifier) @identifier
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    
    variable_usage = {}
    
    for match in matches:
        _, _match = match
        identifier_node = _match['identifier'][0]
        var_name = get_node_text(src, identifier_node)
        identifier_pos = identifier_node.start_byte
        
        # Skip common keywords and built-ins
        if var_name in {'def', 'class', 'if', 'else', 'for', 'while', 'try', 'except', 'import', 'from', 'return', 'self', 'True', 'False', 'None'}:
            continue
        
        if var_name not in variable_usage:
            variable_usage[var_name] = {
                'used_inside_functions_classes': False,
                'used_outside_functions_classes': False,
                'inside_positions': [],
                'outside_positions': []
            }
        
        # Check if this usage is inside any function or class
        is_inside = False
        for start, end in inside_ranges:
            if start < identifier_pos < end:
                is_inside = True
                variable_usage[var_name]['used_inside_functions_classes'] = True
                variable_usage[var_name]['inside_positions'].append(identifier_pos)
                break
        
        if not is_inside:
            variable_usage[var_name]['used_outside_functions_classes'] = True
            variable_usage[var_name]['outside_positions'].append(identifier_pos)
    
    return variable_usage

def get_variables_used_by_functions_classes(code: str) -> list:
    """
    Get global variables that are used by functions and classes.
    Includes variables that are:
    1. Used inside functions/classes but NOT outside (exclusively internal)
    2. Used both inside and outside (shared usage)
    
    Excludes variables that are only used outside functions/classes.
    """
    global_vars = get_global_variables(code)
    variable_usage = analyze_variable_usage(code)
    
    # Extract variable names from global assignments
    global_var_names = set()
    src = bytes(code, "utf-8")
    
    for start, end, text in global_vars:
        # Parse the assignment to get the variable name(s)
        # Handle cases like: x = 1, a, b = 1, 2, etc.
        try:
            # Simple heuristic: get the part before '='
            var_part = text.split('=')[0].strip()
            # Handle multiple assignments: a, b = 1, 2
            var_names = [name.strip() for name in var_part.split(',')]
            global_var_names.update(var_names)
        except:
            continue
    
    # Filter global variables based on usage patterns
    result_vars = []
    
    for start, end, text in global_vars:
        # Get variable names from this assignment
        try:
            var_part = text.split('=')[0].strip()
            var_names = [name.strip() for name in var_part.split(',')]
            
            # Check if any variable in this assignment is used by functions/classes
            should_include = False
            for var_name in var_names:
                if var_name in variable_usage:
                    usage = variable_usage[var_name]
                    # Include if used inside functions/classes (regardless of outside usage)
                    if usage['used_inside_functions_classes']:
                        should_include = True
                        break
            
            if should_include:
                result_vars.append([start, end, text])
                
        except:
            # If parsing fails, include the variable to be safe
            result_vars.append([start, end, text])
    
    return result_vars

def orchestrate_code_extraction_filtered(code: str) -> str:
    """
    Enhanced orchestration function that only includes global variables
    that are used by functions and classes.
    """
    # Extract all components
    functions = get_func_position_info(code)
    classes = get_class_position_info(code)
    # Get only global variables used by functions/classes
    global_vars = get_variables_used_by_functions_classes(code)
    
    # Filter out all nested functions and classes
    top_level_functions, top_level_classes = filter_nested_items(functions, classes, code)
    
    # Combine all components with their positions
    all_components = []
    
    # Add filtered global variables
    for var_info in global_vars:
        all_components.append(('variable', var_info[0], var_info[1], var_info[2]))
    
    # Add top-level functions only
    for func_info in top_level_functions:
        all_components.append(('function', func_info[0], func_info[1], func_info[2]))
    
    # Add top-level classes
    for class_info in top_level_classes:
        all_components.append(('class', class_info[0], class_info[1], class_info[2]))
    
    # Sort by start_byte to maintain original order
    all_components.sort(key=lambda x: x[1])
    
    # Filter to ensure global variables come before functions/classes
    # and reconstruct the code
    result_parts = []
    
    # First pass: collect global variables
    for component_type, start_byte, end_byte, text in all_components:
        if component_type == 'variable':
            result_parts.append(text)
    
    # Second pass: collect functions and classes in order
    for component_type, start_byte, end_byte, text in all_components:
        if component_type in ['function', 'class']:
            result_parts.append(text)
    
    # Join with double newlines for readability
    return '\n\n'.join(result_parts)

def code_excluding_func(code: str, func_name: str) -> str:
    """
    Return the code with the specified top-level function removed.
    Handles both plain and decorated functions.
    """
    functions = get_func_position_info(code)
    classes = get_class_position_info(code)
    top_level_functions, _ = filter_nested_items(functions, classes, code)

    src = bytes(code, "utf-8")
    target_range = None
    print(top_level_functions)
    for start, end, text in top_level_functions:
        # Check the first non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[1] if lines[0].startswith("@") else lines[0]
        # Match the header against target name
        if header.startswith(f"def {func_name}"):
            target_range = (start, end)
            break

    if not target_range:
        # Nothing removed
        return code

    start, end = target_range
    before = src[:start].decode("utf-8")
    after = src[end:].decode("utf-8")

    # Clean formatting
    result = (before.rstrip("\n") + "\n\n" + after.lstrip("\n")).strip("\n") + "\n"
    return result


def get_function_body(code: str, func_name: str) -> str:
    """
    Args:
        code (str): this should be the gfg problem user template, this contains the class name of the problem (if any)
        func_name (str): this is the name of the function that the user is trying to complete
    Returns:
        str: the generated code with the user code inserted into the class
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (_) @func_name (#eq? @func_name "{func_name}")
            body: (_) @func.body
        )
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    body = get_node_text(src, matches[0][1]['func.body'][0])
    return body

def get_function_body(code: str, func_name: str) -> str:
    """
    Args:
        code (str): this should be the gfg problem user template, this contains the class name of the problem (if any)
        func_name (str): this is the name of the function that the user is trying to complete
    Returns:
        str: the generated code with the user code inserted into the class
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (_) @func_name (#eq? @func_name "{func_name}")
            body: (_) @func.body
        )
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    if matches:
        body = get_node_text(src, matches[0][1]['func.body'][0])
        return body
    return None

def get_function_node(code: str, func_name: str) -> str:
    """
    Args:
        code (str): this should be the gfg problem user template, this contains the class name of the problem (if any)
        func_name (str): this is the name of the function that the user is trying to complete
    Returns:
        str: the generated code with the user code inserted into the class
    """
    src = bytes(code, "utf-8")
    tree = PYTHON_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (_) @func_name (#eq? @func_name "{func_name}")
            body: (_) @func.body
        ) @func.full
    """
    extraction_query = PYTHON_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    # print(matches)
    return matches[0][1]['func.full'][0], matches[0][1]['func.body'][0]
 

def python_put_under_class(code: str, user_code: str, func_sign_info: Dict[str, str]) -> str:
    """
    Add all top-level functions and classes in the code to the specified class.
    """
    # Extract all components
    imports = get_python_imports(code)
    if imports:
        imports = imports[0] + '\n'
    else:
        imports = ""
    code = orchestrate_code_extraction_filtered(code)
    class_name, func_sign, _ = func_sign_info
    
    if class_name is None:
        return imports + '\n' + code
    func_name = func_sign[0]
    func_name = func_name.split('(')[0].strip()
    # When class_name is not null but it is in code
    if f"class {class_name}" not in code:
        code_func_body = get_function_body(code, func_name)
        if code_func_body is None:
            return imports + '\n' + code
        other_funcs_code = code_excluding_func(code, func_name)
        src = bytes(user_code, "utf-8")
        tree = PYTHON_PARSER.parse(src)
        query = f"""
            (class_definition
                name: (_) @class.name (#eq? @class.name "{class_name}")
                body: (block
                    (function_definition
                        name: (_) @func_name (#eq? @func_name "{func_name}") 
                        body: (_) @func.body
                    )
                )
            )
        """
        extraction_query = PYTHON_LANG.query(query)
        matches = extraction_query.matches(tree.root_node)
        func_body_node = matches[0][1]['func.body'][0]
        start_line = src[:func_body_node.start_byte].decode('utf-8').splitlines()[-1]
        indent     = ' ' * (len(start_line) - len(start_line.lstrip()))
        new_body = "\n" + indent * 2 + ("\n" + indent).join(code_func_body.rstrip().splitlines()) + '\n'
        src_bytes = bytearray(src)
        src_bytes[func_body_node.start_byte:func_body_node.end_byte] = new_body.encode('utf-8')
        code = src_bytes.decode('utf-8')
    
    return imports + '\n' + code    

########## For executor ###########

if __name__ == '__main__':
    code = """
def foo():
    return 1

def bar():
    return 2
"""
# Excluding "foo" should leave only bar()
#     code = """
# @only_func
# def only_func():
#     return "hello"
    
# def a():
#     return 1
# """
# Excluding "target" should keep global x and keep()

    print(code_excluding_func(code, 'foo'))
    code3 = """
@staticmethod
def decorated_func():
    return "hi"

def normal_func():
    return "ok"
    """

    print(code_excluding_func(code3, "decorated_func"))
