from tree_sitter_language_pack import get_language, get_parser
from typing import Dict, List, Tuple
from tree_sitter import Node
import re
from difflib import SequenceMatcher
JAVA_PARSER, JAVA_LANG = get_parser('java'), get_language('java')
JAVA_BASE_IMPORTS = "import java.util.*;\nimport java.io.*;\nimport java.math.*;\nimport java.lang.*;"
import pickle
import javalang.tree as ast
import javalang.parse as parse

def get_node_text(src, node):
    return src[node.start_byte:node.end_byte].decode('utf-8')

def get_java_imports(code: str) -> List[str]:
    src = bytes(code, 'utf-8')
    tree = JAVA_PARSER.parse(src)
    query = """    
        (package_declaration) @package
        (import_declaration) @import_statement
    """
    import_stmts = ""
    import_lines = []
    package_stmts = ""
    package_lines = []
    query = JAVA_LANG.query(query)
    target = query.matches(tree.root_node)
    if not target:
        return "", ""
    for _match in target:
        _id, _match_dict = _match
        if _id == 0:
            package_lines.append(get_node_text(src, _match_dict['package'][0]))
        if _id == 1:
            import_lines.append(get_node_text(src, _match_dict['import_statement'][0]))
    import_stmts = '\n'.join(import_lines)
    package_stmts = '\n'.join(package_lines)
    return package_stmts + '\n' + import_stmts, package_lines + import_lines

def get_java_imports_list(code: str) -> List[str]:
    src = bytes(code, 'utf-8')
    tree = JAVA_PARSER.parse(src)
    query = """    
        (package_declaration) @package
        (import_declaration) @import_statement
    """
    import_lines = []
    package_lines = []
    query = JAVA_LANG.query(query)
    target = query.matches(tree.root_node)
    if not target:
        return [], []
    for _match in target:
        _id, _match_dict = _match
        if _id == 0:
            package_lines.append(get_node_text(src, _match_dict['package'][0]))
        if _id == 1:
            import_lines.append(get_node_text(src, _match_dict['import_statement'][0]))
    return package_lines, import_lines

def remove_java_imports(code: str) -> Tuple[str, List[str]]:
    package_lines, import_lines = get_java_imports_list(code)
    cleaned = []
    for line in code.split("\n"):
        if not any([pkg_ln in line.strip() for pkg_ln in package_lines]) and not any([import_ln in line.strip() for import_ln in import_lines]) :
            cleaned.append(line)
    cleaned = "\n".join(cleaned)
    return cleaned, import_lines


def normalize_for_comparison(name):
    """Normalize method name for similarity comparison"""
    return name.replace('_', '').lower()

def calculate_similarity(ref_name, method_name):
    """Calculate similarity between normalized names"""
    norm_ref = normalize_for_comparison(ref_name)
    norm_method = normalize_for_comparison(method_name)
    return SequenceMatcher(None, norm_ref, norm_method).ratio()

def map_methods_to_references(extracted_methods, reference_names):
    """
    Map extracted method names to reference names based on similarity.
    Returns a mapping: {original_method_name: reference_name}
    """
    matches = {}
    used_references = set()
    
    # Create similarity matrix
    similarity_matrix = []
    for method in extracted_methods:
        for ref in reference_names:
            sim = calculate_similarity(ref, method)
            similarity_matrix.append((sim, ref, method))
    
    # Sort by similarity (highest first) and assign greedily
    similarity_matrix.sort(key=lambda x: x[0], reverse=True)
    
    for similarity, ref_name, method_name in similarity_matrix:
        if ref_name not in used_references and method_name not in matches:
            matches[method_name] = ref_name
            used_references.add(ref_name)
    
    return matches

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
        for func_node in captures.get("func", []):
            funcs.append((func_node, captures))

    if not funcs:
        return None

    # Pick the function with the max start_byte (latest in file)
    last_func_node, match_dict = max(funcs, key=lambda x: x[0].start_byte)
    return {"func": last_func_node, **match_dict}

def align_func_name(code: str, func_name: str) -> str:
    src = bytes(code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    query = f"""
        (function_definition
            name: (identifier) @name
        ) @func
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    last_func = get_last_function(matches)
    last_func_name_start = last_func['name'][0].start_byte
    last_func_name_end = last_func['name'][0].end_byte
    renamed_full_code = src[:last_func_name_start] + bytes(func_name, "utf-8") + src[last_func_name_end:]
    return renamed_full_code.decode("utf-8")

# def get_last_n_functions(matches, n=1, src_bytes=None):
#     """
#     Return the last n functions (by end_byte) from Tree-sitter matches.
#     Skips function named 'main'.
#     """
#     funcs = []
#     for _, captures in matches:
#         name_nodes = captures.get("name", [])
#         if name_nodes and src_bytes:
#             name_text = src_bytes[name_nodes[0].start_byte:name_nodes[0].end_byte].decode()
#             if name_text == "main":
#                 continue
#         for func_node in captures.get("func", []):
#             funcs.append((func_node, captures))

#     if not funcs:
#         return []

#     funcs.sort(key=lambda x: x[0].end_byte)
#     last_funcs = funcs[-n:]
#     return [{"func": fn, **caps} for fn, caps in last_funcs]
def align_callee(code: str, callee_original_name: str, aligned_func_name: str) -> str:
    src = bytes(code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    query = f"""
        (method_invocation
            name: (_) @callee_name
        )
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    # print("CALLEE")
    # print(matches)
    for match in matches[::-1]:
        _, match = match
        # print("match name", get_node_text(src, match['callee_name'][0]))
        if get_node_text(src, match['callee_name'][0]) == callee_original_name:
            src = src[:match['callee_name'][0].start_byte] + bytes(aligned_func_name, "utf-8") + src[match['callee_name'][0].end_byte:]
    return src

def make_all_methods_public_static(code: str):
    src = bytes(code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    query = f"""
        (method_declaration
            (modifiers)? @modifier
        )
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(tree.root_node)
    # print("CALLEE")
    # print(matches)
    if matches:
        for match in matches[::-1]:
            _, match = match
            # print("match name", get_node_text(src, match['callee_name'][0]))
            src = src[:match['modifier'][0].start_byte] + b"public static " + src[match['modifier'][0].end_byte:]
    return src.decode('utf-8')

def java_polyhumaneval_formatter(code: str, ref_name: List[str]) -> str:
    if "class" not in code:
        code = "public class Global {\n    " + code + "\n}"
    code = remove_java_main(code)
    # 1. Extract all method names from the code
    src = bytes(code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    
    # Get original class name
    class_query = """(class_declaration name: (_) @class_name)"""
    class_extraction_query = JAVA_LANG.query(class_query)
    class_matches = class_extraction_query.matches(tree.root_node)
    original_class_name = get_node_text(src, class_matches[0][1]['class_name'][0]) if class_matches else None
    
    # Extract all method names (excluding main)
    method_query = """(method_declaration name: (_) @method_name) @full_method"""
    method_extraction_query = JAVA_LANG.query(method_query)
    method_matches = method_extraction_query.matches(tree.root_node)
    
    # Filter out main methods and collect method info
    methods_info = []
    for _, match in method_matches:
        method_name = get_node_text(src, match['method_name'][0])
        if method_name != "main":
            methods_info.append({
                'name': method_name,
                'name_node': match['method_name'][0],
                'full_node': match['full_method'][0]
            })
    
    # 2. Map extracted methods to reference names using similarity
    extracted_method_names = [m['name'] for m in methods_info]
    method_mapping = map_methods_to_references(extracted_method_names, ref_name)
    
    # 3. Perform renaming based on mapping
    renamed_full_code = src
    
    # Sort by start_byte in reverse order to avoid offset issues
    methods_info.sort(key=lambda x: x['name_node'].start_byte, reverse=True)
    
    for method_info in methods_info:
        original_name = method_info['name']
        if original_name in method_mapping:
            new_name = method_mapping[original_name]
            
            # Replace method declaration
            start_byte = method_info['name_node'].start_byte
            end_byte = method_info['name_node'].end_byte
            renamed_full_code = (renamed_full_code[:start_byte] + 
                               bytes(new_name, "utf-8") + 
                               renamed_full_code[end_byte:])
            
            # Replace method calls using align_callee
            renamed_full_code = align_callee(
                renamed_full_code.decode("utf-8"), 
                original_name, 
                new_name
            )
            renamed_full_code = bytes(renamed_full_code.decode("utf-8"), "utf-8")
    
    # 4. Update method references (ClassName::methodName)
    if original_class_name:
        method_ref_query = """(method_reference
            (_) @cls
            (_) @method
        ) @method_ref"""
        method_ref_extraction_query = JAVA_LANG.query(method_ref_query)
        tree = JAVA_PARSER.parse(renamed_full_code)
        method_ref_matches = method_ref_extraction_query.matches(tree.root_node)
        
        # Sort by start_byte in reverse order to avoid offset issues
        method_ref_matches.sort(key=lambda x: x[1]['cls'][0].start_byte, reverse=True)
        
        for _, match in method_ref_matches:
            cls_node = match['cls'][0]
            cls_name = get_node_text(renamed_full_code, cls_node)
            if cls_name == original_class_name:
                # Replace class name in method reference
                renamed_full_code = (renamed_full_code[:cls_node.start_byte] + 
                                   bytes('Global', 'utf-8') + 
                                   renamed_full_code[cls_node.end_byte:])

    # 5. Rename class to Global
    if class_matches:
        # Re-parse after method reference updates
        tree = JAVA_PARSER.parse(renamed_full_code)
        class_extraction_query = JAVA_LANG.query(class_query)
        class_matches = class_extraction_query.matches(tree.root_node)
        if class_matches:
            class_name_node = class_matches[0][1]['class_name'][0]
            renamed_full_code = (renamed_full_code[:class_name_node.start_byte] + 
                               bytes('Global', 'utf-8') + 
                               renamed_full_code[class_name_node.end_byte:])
    
    result = renamed_full_code.decode("utf-8")
    
    # 6. Fix constructor names
    if original_class_name:
        result = re.sub(rf'\b{re.escape(original_class_name)}\s*\(', 'Global(', result)
    
    # 7. Remove class modifiers
    result = re.sub(r'(?:\b(?:public|private|protected|final|abstract|static)\s+)+(?=class\b)', '', result)
    
    result = make_all_methods_public_static(result)
    
    return result

########## for Executor ##########
def get_unique_classes(text: str, full_code_provided: bool = False):
    unique_classes = {}
    src = bytes(text, 'utf-8')
    query = """
    (class_declaration
        (modifiers)? @mod
        name: (identifier) @class.name
    ) @class.body
    """
    root = JAVA_PARSER.parse(src).root_node
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    for _match in matches:
        inner_class = False
        _id, _match_dict = _match
        class_name = get_node_text(src, _match_dict['class.name'][0])
        if class_name not in unique_classes:
            if unique_classes:
                for (_, body_node) in unique_classes.values():
                    if body_node.start_byte < _match_dict['class.body'][0].start_byte < body_node.end_byte:
                        inner_class = True
                        break
            if not inner_class:
                if 'mod' in _match_dict:
                    unique_classes[class_name] = (_match_dict['mod'][0], _match_dict['class.body'][0])
                else:
                    unique_classes[class_name] = ("", _match_dict['class.body'][0])
        else:
            cur_class_start_byte = _match_dict['class.body'][0].start_byte
            cur_class_end_byte = _match_dict['class.body'][0].end_byte
            stored_class_node = unique_classes[class_name][1]
            stored_class_start_byte = stored_class_node.start_byte
            stored_class_end_byte = stored_class_node.end_byte
            if full_code_provided:
                if cur_class_start_byte < stored_class_start_byte:
                    if 'mod' in _match_dict:
                        unique_classes[class_name] = (_match_dict['mod'][0], _match_dict['class.body'][0])
                    else:
                        unique_classes[class_name] = ("", _match_dict['class.body'][0])
                else:
                    pass
            else:
                if cur_class_start_byte > stored_class_start_byte:
                    if 'mod' in _match_dict:
                        unique_classes[class_name] = (_match_dict['mod'][0], _match_dict['class.body'][0])
                    else:
                        unique_classes[class_name] = ("", _match_dict['class.body'][0])
                else:
                    pass
    
    return_dict = {}
    import tree_sitter
    for cls_name, (mod_node, body_node) in unique_classes.items():
        if isinstance(mod_node, tree_sitter.Node):
            return_dict[cls_name] = (get_node_text(src, mod_node), get_node_text(src, body_node))
        else:
            return_dict[cls_name] = ("", get_node_text(src, body_node))
    return return_dict

def remove_modifier(code: str, class_name: str, modifier: str) -> str:
    query = f"""
    (class_declaration
        (modifiers)? @mod
        name: (identifier) @cls_name (#eq? @cls_name "{class_name}")
    )
    """
    src = bytes(code, 'utf-8')
    root = JAVA_PARSER.parse(src).root_node
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    # mod_node = matches[0][1]['mod'][0]
    pos = src.decode('utf-8').find(modifier)
    code_str_list = list(code)
    code_str_list[pos:pos+len(modifier)] = ''
    return ''.join(code_str_list).strip()

def remove_public_if_no_main(code_dict: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    parsed_dict = {}
    for cls_name, (modifier, full_body) in code_dict.items():
        if modifier != 'public':
            parsed_dict[cls_name] = (modifier, full_body)
            continue
        has_main = False
        src = bytes(full_body, 'utf-8')
        tree = JAVA_PARSER.parse(src)
        root = tree.root_node
        query = """
            (class_declaration
                name: (identifier) @cls_name
                body: (class_body
                    (method_declaration
                        name: (identifier) @name (#eq? @name "main")
                    ) 
                )
            ) 
        """
        extraction_query = JAVA_LANG.query(query)
        matches = extraction_query.matches(root)
        for _match in matches:
            _id, _match_dict = _match
            if _match_dict['name'][0].text == "main":
                has_main = True
                break
        if has_main:
            parsed_dict[cls_name] = (modifier, full_body)
        else:
            modified_body = remove_modifier(full_body, cls_name, modifier)
            
            parsed_dict[cls_name] = ("", modified_body)
    return parsed_dict

def find_script_name(code: str) -> str:
    src = bytes(code, 'utf-8')
    tree = JAVA_PARSER.parse(src)
    query = """
    (class_declaration
        name: (identifier) @cls_name
        body: (class_body
            (method_declaration
                (modifiers) @modifier (#eq? @modifier "public static")
                name: (identifier) @name (#eq? @name "main")
            ) 
        )
    )
    """
    root = tree.root_node
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    for _match in matches:
        _id, _match_dict = _match
        return get_node_text(src, _match_dict['cls_name'][0])
    
def seperate_java_files(code_info_dict: dict) -> str:
    code_info_dict = remove_public_if_no_main(code_info_dict)
    filename_dict = {}
    ordinary_code = ""
    for cls_name, (modifier, code) in code_info_dict.items():
        if modifier.find("public") != -1:
            filename_dict[f'{cls_name}.java'] = code
        else:
            ordinary_code += '\n' + code
    if len(filename_dict) == 1:
        filename_dict['Temp.java'] = ordinary_code
    elif len(filename_dict) == 0:
        script_name = find_script_name(ordinary_code)
        if script_name:
            filename_dict[f'{script_name}.java'] = ordinary_code
    return filename_dict

#### Test Generation #######
def parse_param_declaration_from_method_code(method_code: str):
    """
    Analyze method parameters' types and names
    :param method_code: input method, usually focal method
    :return: a dict in which the keys are parameter names, and the values are corresponding types.
    """
    
    params = {}
    tmp_method_code = "public class TmpClass {\n" + method_code + "}\n"
    # print(tmp_method_code)
    tree = JAVA_PARSER.parse(bytes(tmp_method_code, "utf-8"))
    method_param_query = JAVA_LANG.query(
        """
        (class_declaration 
            body: (class_body
                (method_declaration 
                    parameters: (formal_parameters
                        (formal_parameter 
                            type: (_) @type_identifier
                            name: (identifier) @param_name )
                        )
                )
            )
        )
        (class_declaration 
            body: (class_body
                (method_declaration 
                    parameters: (formal_parameters
                    (_ 
                        (type_identifier) @type_identifier
                        (variable_declarator name: (_) @param_name)
                    )
                    )
                )
            )
        )
        """
    )
    res = method_param_query.captures(tree.root_node)
    """
    This is for the new version of tree-sitter-java
    """
    try:
        for type_iden, param_name in zip(res['type_identifier'], res['param_name']):
            params[str(param_name.text, encoding="utf-8")] = str(
                type_iden.text, encoding="utf-8"
            )
    except Exception as e:
        return params
    
    """
    This is for the old version of tree-sitter-java
    """
    return params

def parse_fields_from_class_code(class_str: str, need_prefix=True):
    """
    Analyze defined fields for given class.
    :param class_str: class code in a string.
    :return: list of field dicts, for eaxmple:
            {
                "field_name": field_name,
                "field_type": field_type,
                "field_modifiers": field_modifiers,
                "declaration_text": declaration_text,
            }
    """
    parser = get_parser('java')
    tmp_class_str = pickle.loads(pickle.dumps(class_str))
    if need_prefix:
        tmp_class_str = "public class TmpClass{\n" + class_str
    tree = parser.parse(bytes(tmp_class_str, "utf-8"))
    rets = []

    field_decl_query = JAVA_LANG.query(
        """
        (field_declaration 
            type: (_) @type_name 
            declarator: (variable_declarator name: (identifier)@var_name)
        ) @field_decl
        """
    )

    fields = field_decl_query.captures(tree.root_node)
    if len(fields) % 3 != 0:
        if int(len(fields) / 3) == 0:
            return []
        else:
            fields = fields[: -(len(fields) % 3)]
    num_iter = len(fields) / 3
    for _ in range(int(num_iter)):
        field_name = ""
        field_type = ""
        field_modifiers = "deprecated"
        declaration_text = ""

        """
        This is for the new version of tree-sitter-java
        """
        
        for query_name, node_list in fields.items():
            if query_name == "field_decl":
                declaration_text = str(node_list[0].text, encoding="utf-8")
            elif query_name == "type_name":
                field_type = str(node_list[0].text, encoding="utf-8")
            elif query_name == "var_name":
                field_name = str(node_list[0].text, encoding="utf-8")
            else:
                raise NotImplementedError(f"Unknown query result name {query_name}")
        
        if (
                field_name != ""
                and field_modifiers != ""
                and field_type != ""
                and declaration_text != ""
        ):
            rets.append(
                {
                    "field_name": field_name,
                    "field_type": field_type,
                    "field_modifiers": field_modifiers,
                    "declaration_text": declaration_text,
                }
            )

    return rets

def is_method_public(java_code):
    # Parse the Java code
    java_code = 'class Test {\n' + java_code.replace('\r\n', '\n\n') + '\n}'
    tree = parse.parse(java_code)

    # Search for the method
    for path, node in tree.filter(ast.MethodDeclaration):
        if node:
            for modifier in node.modifiers:
                if modifier == 'public':
                    return True
                elif modifier in ['private', 'protected']:
                    return False
    return False

def remove_java_main(code: str) -> str:
    src = bytes(code, 'utf-8')
    root = JAVA_PARSER.parse(src).root_node
    query = """
        (class_declaration
            name: (identifier) @cls_name
            body: (class_body
                (method_declaration
                    name: (identifier) @name (#eq? @name "main")
                ) @method
            )
        ) @cls
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    main_code_pos = []
    for _match in matches:
        _, _match_dict = _match
        _node = _match_dict['method'][0]
        main_code_pos.append([_node.start_byte, _node.end_byte])
    
    main_code_pos.sort(key=lambda x: x[0], reverse=True)
    
    main_removed_code = src
    
    for pos in main_code_pos:
        start, end = pos
        main_removed_code = main_removed_code[:start] + bytes("", "utf-8") + main_removed_code[end:]
    
    code = main_removed_code.decode('utf-8')
    cleaned_lines = []
    for line in code.split("\n"):
        if line.strip() == "":
            continue
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)    

def remove_public_if_no_main(code_dict):
    parsed_dict = {}
    for cls_name, (modifier, full_body) in code_dict.items():
        if modifier != 'public':
            parsed_dict[cls_name] = (modifier, full_body)
            continue
        has_main = False
        src = bytes(full_body, 'utf-8')
        parser = get_parser('java')
        lang = get_language('java')
        root = parser.parse(src).root_node
        query = """
            (class_declaration
                name: (identifier) @cls_name
                body: (class_body
                    (method_declaration
                        name: (identifier) @name (#eq? @name "main")
                    ) 
                )
            ) 
        """
        extraction_query = lang.query(query)
        matches = extraction_query.matches(root)
        for _match in matches:
            _id, _match_dict = _match
            if _match_dict['name'][0].text == "main":
                has_main = True
                break
        if has_main:
            parsed_dict[cls_name] = (modifier, full_body)
        else:
            modified_body = remove_modifier(full_body, cls_name, modifier)
            
            parsed_dict[cls_name] = ("", modified_body)
    return parsed_dict

### hackerrank code gen
def get_func_sign_details(func_sign: str) -> str:
    """
    Get the details of the function signature.
    List all possible func_signature that can help us locate the class
    """
    src = bytes(func_sign, "utf-8")
    tree = JAVA_PARSER.parse(src)
    root = tree.root_node
    query = """
    (method_declaration
        (modifiers)? @mod
        type: (_) @type
        name: (_) @name
        parameters: (_) @params
    )
    """
    _modifier, _type, _name, _params = None, None, None, None
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    for _match in matches:
        _id, _match = _match
        if _id == 0:
            # Safely extract text with None checking
            _modifier = _match['mod'][0].text if 'mod' in _match and _match['mod'] else None
            _type = _match['type'][0].text if _match['type'][0].text else None
            _name = _match['name'][0].text if _match['name'][0].text else None
            _params = _match['params'][0].text if _match['params'][0].text else None
    
    if _name is not None:
        return _name.decode('utf-8')
    return ""
    
def correct_cls_name(cls_code: str, cls_name: str, func_name: str) -> str:
    """
    Correct the class name if it is miswritten.
    """
    src = bytes(cls_code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    root = tree.root_node

    query = f"""
    (class_declaration
        name: (_) @class_name
        body: (
                (class_body
                    (method_declaration
                        name: (_) @method_name (#eq? @method_name "{func_name}")
                    )
            )
        )
    )
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    for _match in matches:
        _id, _match = _match
        if _id == 0:
            class_name_text = _match['class_name'][0].text
            if class_name_text is not None:
                _cls_name = class_name_text.decode('utf-8')
                if _cls_name != cls_name:
                    cls_code = cls_code.replace(_cls_name, cls_name)
    return cls_code

def ensure_script_name_class_public(code: str, script_name: str) -> str:
    """
    Correct the class name if it is miswritten.
    """
    src = bytes(code, "utf-8")
    tree = JAVA_PARSER.parse(src)
    root = tree.root_node
    query = f"""
    (class_declaration
        (modifiers)? @mod
        name: (_) @class_name (#eq? @class_name "{script_name}") 
    ) @full_cls
    """
    extraction_query = JAVA_LANG.query(query)
    matches = extraction_query.matches(root)
    match = matches[0][1]
    start = match['full_cls'][0].start_byte

    if 'mod' in match and match['mod']:
        mod_text_raw = match['mod'][0].text
        if mod_text_raw is not None:
            mod_text = mod_text_raw.decode('utf-8')
            if 'public' not in mod_text.split():
                src = src[:start] + b'public ' + src[start:]
        else:
            src = src[:start] + b'public ' + src[start:]
    else:
        src = src[:start] + b'public ' + src[start:]
        
    return src.decode('utf-8')

def clean_hackerrank_java_code(code: str, driver_code: Tuple[str, str], java_func_sign_info: Dict[str, str]) -> str:
    """
    Clean the java code for hackerrank.
    - class_name for preventing the to be written class being written as public and to correct the miswritten class_name containing func_sign
    - func_sign is to help locate the class containing it
    - script_name ensures only all classes called the script name to be removed
    """
    class_name, func_sign, script_name = java_func_sign_info
    assert len(func_sign) == 1, "only questions with one function signature are supported"
    func_sign = func_sign[0]
    template_head, template_tail = driver_code

    # this ease the change of the code
    if class_name:
        
        import_lines = get_java_imports(code)
        if import_lines:
            import_lines = import_lines[0]
        else:
            import_lines = ""
            
        code = remove_java_main(code)
        classes = get_unique_classes(code, False)
        func_name = get_func_sign_details(func_sign)
        new_classes = []
        for cls_name, (modifier, cls_code) in classes.items():
            # Check if modifier is valid and contains 'public'
            if modifier and 'public' in modifier:
                cls_code = cls_code.replace(modifier + f' class {cls_name}', f'class {cls_name}')
            # Check if func_name is valid before using in string operations
            if func_name and func_name in cls_code:
                if class_name:
                    cls_code = correct_cls_name(cls_code, class_name, func_name)
            new_classes.append(cls_code)
        driver_code = template_tail
        driver_imports = get_java_imports(driver_code)[1]
        non_import_driver_lines = []
        for line in driver_code.split("\n"):
            if line in driver_imports:
                continue
            non_import_driver_lines.append(line)
        driver_code = '\n'.join(non_import_driver_lines)
        driver_code = ensure_script_name_class_public(driver_code, script_name)
        return import_lines + '\n' + JAVA_BASE_IMPORTS + '\n' + '\n'.join(new_classes) + '\n' + driver_code

    else:
        full_code = template_head + '\n' + code + template_tail
        return JAVA_BASE_IMPORTS + '\n' + full_code
        
def clean_geeksforgeeks_java_code(code: str, driver_code: str, java_func_sign_info: Dict[str, str]) -> str:
    """
    Clean the java code for hackerrank.
    - class_name for preventing the to be written class being written as public and to correct the miswritten class_name containing func_sign
    - func_sign is to help locate the class containing it
    - script_name ensures only all classes called the script name to be removed
    """
    class_name, func_sign, script_name = java_func_sign_info
    assert len(func_sign) == 1, "only questions with one function signature are supported"
    func_sign = func_sign[0]
    template_tail = driver_code

    import_lines = get_java_imports(code)
    if import_lines:
        import_lines = import_lines[0]
    else:
        import_lines = ""
        
    code = remove_java_main(code)
    classes = get_unique_classes(code, False)
    func_name = get_func_sign_details(func_sign)
    new_classes = []
    for cls_name, (modifier, cls_code) in classes.items():
        # Check if modifier is valid and contains 'public'
        if modifier and 'public' in modifier:
            cls_code = cls_code.replace(modifier + f' class {cls_name}', f'class {cls_name}')
        # Check if func_name is valid before using in string operations
        if func_name and func_name in cls_code:
            if class_name:
                cls_code = correct_cls_name(cls_code, class_name, func_name)
        new_classes.append(cls_code)
    driver_code = template_tail
    driver_imports = get_java_imports(driver_code)[1]
    non_import_driver_lines = []
    for line in driver_code.split("\n"):
        if line in driver_imports:
            continue
        non_import_driver_lines.append(line)
    driver_code = '\n'.join(non_import_driver_lines)
    driver_code = ensure_script_name_class_public(driver_code, script_name)
    return import_lines + '\n' + JAVA_BASE_IMPORTS + '\n' + '\n'.join(new_classes) + '\n' + driver_code

if __name__ == '__main__':
    # code = "import java.util.List;\n\npublic class Solution {\n    public static int nonDivisibleSubset(int k, List<Integer> s) {\n        int[] freq = new int[k];\n        for (int val : s) {\n            freq[val % k]++;\n        }\n\n        int count = 0;\n\n        // Include at most one element with remainder 0\n        if (freq[0] > 0) count++;\n\n        // For each pair of remainders r and k - r, take the larger group\n        for (int r = 1; r <= k / 2; r++) {\n            if (r == k - r) {\n                // When k is even and r == k/2, include at most one\n                if (freq[r] > 0) count++;\n            } else {\n                count += Math.max(freq[r], freq[k - r]);\n            }\n        }\n\n        return count;\n    }\n}"
    # # print("TYPE:\n", type(code))
    # # print("BEFORE:\n", code)
    # code = clean_hackerrank_java_code(code, "public class Solution {\n    public static void main(String[] args) throws IOException {\n        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));\n        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(System.getenv(\"OUTPUT_PATH\")));\n\n        String[] firstMultipleInput = bufferedReader.readLine().replaceAll(\"\\\\s+$\", \"\").split(\" \");\n\n        int n = Integer.parseInt(firstMultipleInput[0]);\n\n        int k = Integer.parseInt(firstMultipleInput[1]);\n\n        String[] sTemp = bufferedReader.readLine().replaceAll(\"\\\\s+$\", \"\").split(\" \");\n\n        List<Integer> s = new ArrayList<>();\n\n        for (int i = 0; i < n; i++) {\n            int sItem = Integer.parseInt(sTemp[i]);\n            s.add(sItem);\n        }\n\n        int result = Result.nonDivisibleSubset(k, s);\n\n        bufferedWriter.write(String.valueOf(result));\n        bufferedWriter.newLine();\n\n        bufferedReader.close();\n        bufferedWriter.close();\n    }\n}\n", ("Result", ["public static int nonDivisibleSubset(int k, List<Integer> s)"], "Solution"))
    # print("AFTER:\n", code)
    code = "public class ElectronicsShop {\n    public static int getMoneySpent(int[] keyboards, int[] drives, int b) {\n        // Sort both arrays to use a two-pointer approach\n        java.util.Arrays.sort(keyboards); // ascending\n        java.util.Arrays.sort(drives);    // ascending\n\n        int best = -1;\n        int j = drives.length - 1; // start from the most expensive drive\n\n        for (int i = 0; i < keyboards.length; i++) {\n            // If even the cheapest drive with this keyboard exceeds budget, larger keyboards will also exceed\n            if (keyboards[i] + drives[0] > b) break;\n\n            // Move j left until the sum fits within the budget\n            while (j >= 0 && keyboards[i] + drives[j] > b) {\n                j--;\n            }\n\n            // If no drive can pair with this (or any more expensive) keyboard, we can stop\n            if (j < 0) break;\n\n            int sum = keyboards[i] + drives[j];\n            if (sum > best) best = sum;\n        }\n\n        return best;\n    }\n\n    public static void main(String[] args) {\n        // Example usage\n        int[] keyboards = {40, 50, 60};\n        int[] drives = {5, 8, 12};\n        int b = 60;\n\n        int maxSpend = getMoneySpent(keyboards, drives, b);\n        System.out.println(\"Max spend: \" + maxSpend);\n    }\n}", "import java.util.Arrays;\n\npublic class MoneySpent {\n    public static int getMoneySpent(int[] keyboards, int[] drives, int b) {\n        // Sort both arrays to use a two-pointer approach\n        java.util.Arrays.sort(keyboards); // ascending\n        java.util.Arrays.sort(drives);    // ascending\n\n        int best = -1;\n        int j = drives.length - 1; // start from the most expensive drive\n\n        for (int i = 0; i < keyboards.length; i++) {\n            // If even the cheapest drive with this keyboard exceeds budget, larger keyboards will also exceed\n            if (keyboards[i] + drives[0] > b) break;\n\n            // Move j left until the sum fits within the budget\n            while (j >= 0 && keyboards[i] + drives[j] > b) {\n                j--;\n            }\n\n            // If no drive can pair with this (or any more expensive) keyboard, we can stop\n            if (j < 0) break;\n\n            int sum = keyboards[i] + drives[j];\n            if (sum > best) best = sum;\n        }\n\n        return best;\n    }\n\n    public static void main(String[] args) {\n        // Example usage (optional demonstration)\n        int[] keyboards = {40, 50, 60};\n        int[] drives = {5, 8, 12};\n        int budget = 60;\n        int result = getMoneySpent(keyboards, drives, budget);\n        System.out.println(\"Money spent: \" + result);\n    }\n}", "import java.util.Arrays;\n\npublic class MoneySpent {\n    public static int getMoneySpent(int[] keyboards, int[] drives, int b) {\n        Arrays.sort(keyboards);\n        Arrays.sort(drives);\n\n        int best = -1;\n        int j = drives.length - 1;\n\n        for (int i = 0; i < keyboards.length; i++) {\n            // If even the cheapest drive with this keyboard exceeds budget, larger keyboards will also exceed\n            if (keyboards[i] + drives[0] > b) break;\n\n            // Move j left until the sum fits within the budget\n            while (j >= 0 && keyboards[i] + drives[j] > b) {\n                j--;\n            }\n\n            // If no drive can pair with this (or any more expensive) keyboard, we can stop\n            if (j < 0) break;\n\n            int sum = keyboards[i] + drives[j];\n            if (sum > best) best = sum;\n        }\n\n        return best;\n    }\n\n    public static void main(String[] args) {\n        int[] keyboards = {40, 50, 60};\n        int[] drives = {5, 8, 12};\n        int budget = 60;\n        int result = getMoneySpent(keyboards, drives, budget);\n        System.out.println(\"Max spend: \" + result);\n    }\n}"
    print(clean_hackerrank_java_code(code, "    private static final Scanner scanner = new Scanner(System.in);\n\n    public static void main(String[] args) throws IOException {\n        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(System.getenv(\"OUTPUT_PATH\")));\n\n        String[] bnm = scanner.nextLine().split(\" \");\n        scanner.skip(\"(\\r\\n|[\\n\\r\\u2028\\u2029\\u0085])*\");\n\n        int b = Integer.parseInt(bnm[0]);\n\n        int n = Integer.parseInt(bnm[1]);\n\n        int m = Integer.parseInt(bnm[2]);\n\n        int[] keyboards = new int[n];\n\n        String[] keyboardsItems = scanner.nextLine().split(\" \");\n        scanner.skip(\"(\\r\\n|[\\n\\r\\u2028\\u2029\\u0085])*\");\n\n        for (int keyboardsItr = 0; keyboardsItr < n; keyboardsItr++) {\n            int keyboardsItem = Integer.parseInt(keyboardsItems[keyboardsItr]);\n            keyboards[keyboardsItr] = keyboardsItem;\n        }\n\n        int[] drives = new int[m];\n\n        String[] drivesItems = scanner.nextLine().split(\" \");\n        scanner.skip(\"(\\r\\n|[\\n\\r\\u2028\\u2029\\u0085])*\");\n\n        for (int drivesItr = 0; drivesItr < m; drivesItr++) {\n            int drivesItem = Integer.parseInt(drivesItems[drivesItr]);\n            drives[drivesItr] = drivesItem;\n        }\n\n        /*\n         * The maximum amount of money she can spend on a keyboard and USB drive, or -1 if she can't purchase both items\n         */\n\n        int moneySpent = getMoneySpent(keyboards, drives, b);\n\n        bufferedWriter.write(String.valueOf(moneySpent));\n        bufferedWriter.newLine();\n\n        bufferedWriter.close();\n\n        scanner.close();\n    }\n}\n", java_func_sign_info= (None, ["static int getMoneySpent(int[] keyboards, int[] drives, int b)"], "Solution", )))
    # print(get_func_sign_details("public static List<Integer> serviceLane(int n, List<List<Integer>> cases)"))
#     code = """static class CatAndMouse {
#     public static String catAndMouse(int x, int y, int z) {
#         int distA = Math.abs(z - x);
#         int distB = Math.abs(z - y);

#         if (distA < distB) {
#             return "Cat A";
#         } else if (distB < distA) {
#             return "Cat B";
#         } else {
#             return "Mouse C";
#         }
#     }
# }"""
#     print(ensure_script_name_class_public(code, "CatAndMouse"))