import os
from tqdm import tqdm
import json
import cProfile
import pickle
from .data import LLM4UTData, SymPromptData
from .utils.prompt_line_utils import *
from extractors.tree_sitter_extraction_utils.java_tree_sitter_utils import parse_param_declaration_from_method_code, parse_fields_from_class_code,is_method_public
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset

# borrowed from LLM4UT's generate_prompts
PROFILER = cProfile.Profile()
FOCAL_METHOD_KEY = "source:source_method_code_format"
FOCAL_CLASS_FIELDS_KEY = "source_class_fields"
FOCAL_METHOD_NAME_KEY = "source:source_method_name"
FOCAL_METHOD_SIGNATURE_KEY = "source:source_method_signature"
FOCAL_CLASSS_OTHER_METHODS_KEY = "source:source_other_method_signature"
FOCAL_CLASS_CONSTRUCTOR_KEY = "content:source_class_constructors"
FOCAL_CLASS_NAME_KEY = "content:source_class_name"
FOCAL_CLASS_CODE_KEY = "content:source_class_code_format"
SOURCE_METHOD_TYPE_CONSTRUCTOR_KEY = "content:parameter_class_constructors"
SOURCE_METHOD_PARAMETER_KEY = "content:parameter_list"
PARAMETER_CLASSES_KEY = "content:parameter_class_signature"
SOURCE_METHOD_SIGNATURE_KEY = "content:source_method_signature"
SOURCE_CLASS_IMPORTS_KEY = "content:source_class_code_imports"
# for LLM4UT deprecated bugs
DEPRECATED_BUGS = [
    "Cli_6",
    "Closure_63",
    "Closure_93",
    "Lang_2",
    "Time_32"
] + [f"Collections_{i}" for i in range(1, 25)]

def get_prompt_info(data, template_category, template_id):
    """
    Get the prompt information from the data
    
    Referenced from https://github.com/LeonYang95/LLM4UT/blob/de09a847ceb81fb1b73514f7c8af3b19a307c7f4/utils/prompt_formats/prompt_formatter.py#L235-L344
    
    """
    focal_method = data[FOCAL_METHOD_KEY]
    focal_class_name = data[FOCAL_CLASS_NAME_KEY]
    focal_class = data[FOCAL_CLASS_CODE_KEY]
    focal_method_signature = data[FOCAL_METHOD_SIGNATURE_KEY]
    focal_class_signature = '.'.join(focal_method_signature.split('#')[:2])
    focal_class_imports = data[SOURCE_CLASS_IMPORTS_KEY]
    
    param_dict = parse_param_declaration_from_method_code(focal_method)
    parameters = [k for k, v in param_dict.items()]
    param_types = [v for k, v in param_dict.items()]
    collected_parameter_classes = list(set([
        item.split('|')[0].replace('#', '.') for item in data[PARAMETER_CLASSES_KEY]
    ]))
    parameter_classes = []
    if len(collected_parameter_classes) == 0 and len(param_types) != 0:
        parameter_classes = pickle.loads(pickle.dumps(param_types))
    elif len(collected_parameter_classes) != 0:
        assert len(param_types) != 0
        for type in param_types:
            found = False
            for class_sig in collected_parameter_classes:
                class_sig = class_sig.strip()
                if class_sig.endswith('.' + type):
                    parameter_classes.append(class_sig)
                    found = True
                    break
            if not found:
                parameter_classes.append(type)
    
    focal_class_fields = [x.strip() for x in data[
        FOCAL_CLASS_FIELDS_KEY]] if FOCAL_CLASS_FIELDS_KEY in data.keys() else parse_fields_from_class_code(
        focal_class)

    focal_class_other_methods = [x.strip() for x in data[
        FOCAL_CLASSS_OTHER_METHODS_KEY]] if FOCAL_CLASSS_OTHER_METHODS_KEY in data.keys() else []
    focal_class_constructor = [x.strip() for x in data[
        FOCAL_CLASS_CONSTRUCTOR_KEY]] if FOCAL_CLASS_CONSTRUCTOR_KEY in data.keys() else ""

    source_method_parameter_class_constructors = [x.strip() for x in data[
        SOURCE_METHOD_TYPE_CONSTRUCTOR_KEY]] if SOURCE_METHOD_TYPE_CONSTRUCTOR_KEY in data.keys() else []
    
    metadata = {
        "focal_method": focal_method,
        "params_lines": get_parameters(parameters, template_category, template_id),
        "param_classes_lines": get_parameter_classes(parameter_classes, template_category, template_id),
        "parameter_class_constructors_lines": get_parameter_class_constructors(source_method_parameter_class_constructors, template_category, template_id),
        "focal_class_constructors_lines": get_focal_class_constructor(focal_class_constructor, focal_class_signature, template_category, template_id),
        "class_fields_lines": get_focal_class_field(focal_class_fields, template_category, template_id),
        "other_methods_lines": get_focal_class_other_method(focal_class_other_methods, template_category, template_id) # this was ignored in the LLM4UT Paper
    }
    
    is_public = is_method_public(focal_method)
    
    return is_public, metadata

    
class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.test_data = []
    
    def load_data(self):
        if self.dataset == 'llm4ut':
            return self.load_llm4ut_data() 
        elif self.dataset == 'symprompt':
            return self.load_symprompt_data()
        else:
            raise NotImplementedError(f"Project is not supported")
    
    def load_symprompt_data(self):
        ds = load_dataset("Code-TREAT/unit_test_generation")
        full_data = ds['test']
        for data in full_data:
            data_id = data["prompt_id"]
            project_name = data['project']
            module_name = data['module']
            class_name = data['class']
            method_name = data['method']
            focal_method = data['focal_method_txt']
            focal_method_globals = data['globals']
            type_context = data['type_context']
            d_uid = f"{project_name}_{module_name}_{class_name}_{method_name}"
            _data = SymPromptData(
                'symprompt', 
                'python', 
                d_uid, 
                data_id, 
                project_name, 
                module_name, 
                class_name,
                method_name,
                focal_method,
                focal_method_globals, 
                type_context
            )
            self.test_data.append(_data)
        return self.test_data

    def load_d4j_data(self, model_type='api'):
        # formatter = PromptFormatter()
        data = []
        code_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLM4UT")
        backup_file = os.path.join(code_base, "data/prompts/source_data.jsonl")
        if os.path.exists(backup_file):
            with open(backup_file, "r", encoding="utf-8") as reader:
                for line in reader.readlines():
                    d = json.loads(line.strip())
                    data.append(d)
        print(len(data))
        for idx, instance in enumerate(data):
            if model_type.lower() == "api":
                new_inst = {}
                is_public = False
                bug_id = instance["extra:project_name"]
                if any(bug_id.startswith(deprecated_bug_id) for deprecated_bug_id in DEPRECATED_BUGS):
                    continue
                new_inst["id"] = bug_id
                # new_inst["strategy"] = strategy
                # new_inst["format"] = format
                # new_inst["ablation"] = ignore_feature
                new_inst["focal_method"] = instance[FOCAL_METHOD_KEY]
                # not important, just for computing the is_public
                is_public, new_inst["metainfo"] = get_prompt_info(instance, ["direct"], 1)
                # print(is_public)
                new_inst["class_name"] = bug_id
                new_inst["method_signature"] = instance['source:source_method_signature']
                new_inst["is_public"] = "1" if is_public else "0"
                new_inst["role"] = "user"
                instance_uid = f"{bug_id}_{instance['source:source_method_signature']}"
                if is_public:
                    _data = LLM4UTData(
                        'llm4ut',
                        'java',
                        instance_uid,
                        bug_id,
                        instance
                    )
                    self.test_data.append(_data)
            elif model_type.lower() == "model":
                raise NotImplementedError("Model type is not implemented.")
            else:
                raise NotImplementedError("only api type is implmented")
        return self.test_data

if __name__ == "__main__":
    data_loader = DataLoader('symprompt', 'python')
    data = data_loader.load_symprompt_data()