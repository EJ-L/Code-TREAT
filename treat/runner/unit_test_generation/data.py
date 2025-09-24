class Data:
    def __init__(self, dataset_name, language, _id):
        self.dataset_name = dataset_name
        self.language = language
        self.key = _id

class LLM4UTData(Data):
    def __init__(self, dataset_name, language, unique_id, bug_id, instance):
        super().__init__(dataset_name, language, unique_id)
        self.instance = instance
        self.bug_id = bug_id
        
class SymPromptData(Data):
    def __init__(self, dataset_name, language, unique_id, id, project, module_name, class_name, method_name, focal_method, globals, type_context):
        super().__init__(dataset_name, language, unique_id)
        self.id = id
        self.project = project
        self.class_name = class_name
        self.module_name = module_name
        self.method_name = method_name
        self.focal_method = focal_method
        self.globals = globals
        self.type_context = type_context