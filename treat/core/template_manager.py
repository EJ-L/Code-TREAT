import json
import os
from typing import Optional, Dict, Any, Tuple, List


class Template:
    """Simple Template class that holds template data and can generate prompts"""
    
    def __init__(self, task_type: str, template_id: int, template_string: str, 
                 system_prompt: str, category: str):
        self.task_type = task_type
        self.template_id = template_id
        self.template_string = template_string
        self.system_prompt = system_prompt
        self.category = category

    def fill_template(self, **kwargs) -> str:
        """Fill the template string with provided parameters"""
        try:
            return self.template_string.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing template parameter {e}")
            # Try to replace manually for special cases
            filled_template = self.template_string
            for key, value in kwargs.items():
                filled_template = filled_template.replace(f"{{{key}}}", str(value))
            return filled_template

    def get_prompt(self, model, openrouter=False, **template_info) -> Tuple[List[Dict], str]:
        """Generate the complete prompt with system message and user query"""
        
        # Fill the user template
        query = self.fill_template(**template_info)
        
        # Build message structure based on model type
        if not openrouter:
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
        else:
            # OpenRouter format
            message = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": query}]
                }
            ]
        
        return message, query

    def __str__(self):
        return f"Template ID: {self.template_id}\nTemplate String: {self.template_string}\nCategory: {self.category}"


class TemplateManager:
    """TemplateManager that uses struct: task -> category -> {system_prompt, user_prompt}"""
    
    def __init__(self, prompt_category: List[str], 
                 json_file_path: str = "templates/prompts.json"):
        self.prompt_category = prompt_category
        self.json_path = json_file_path
        self._templates_cache = {}
        
    def _load_template_data(self) -> Dict[str, Any]:
        """Load the new structured template JSON file"""
        try:
            if not os.path.isabs(self.json_path):
                current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                full_path = os.path.join(current_dir, self.json_path)
            else:
                full_path = self.json_path
                
            with open(full_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Warning: Template file {self.json_path} not found")
            return {}
    
    def load_templates(self, task_type: str, language: str = None, dataset: str = None) -> List[Template]:
        """Load templates for a specific task, with optional language filtering and dataset-specific template mapping"""
        
        # Handle dataset-specific template mapping for vulnerability detection
        effective_task_type = self._get_effective_task_type(task_type, dataset)
        
        cache_key = f"{effective_task_type}_{language}_{'-'.join(sorted(self.prompt_category))}"
        if cache_key in self._templates_cache:
            return self._templates_cache[cache_key]
        
        template_data = self._load_template_data()
        templates = []
        
        if effective_task_type not in template_data:
            print(f"Warning: Task type '{effective_task_type}' not found in template data")
            return templates
        
        task_config = template_data[effective_task_type]
        
        # Handle different structures based on whether it has language sub-keys
        if self._has_language_structure(task_config):
            # Structure: task -> language -> category -> {system_prompt, user_prompt}
            if not language:
                print(f"Warning: Language required for task '{task_type}' but not provided")
                return templates
                
            if language not in task_config:
                print(f"Warning: Language '{language}' not found for task '{task_type}'")
                return templates
            
            language_config = task_config[language]
            
            # Process categories within language
            for category in self.prompt_category:
                if category in language_config:
                    category_data = language_config[category]
                    templates.extend(self._create_templates_from_category(
                        task_type, category, category_data
                    ))
        else:
            # Structure: task -> category -> {system_prompt, user_prompt}
            for category in self.prompt_category:
                if category in task_config:
                    category_data = task_config[category]
                    templates.extend(self._create_templates_from_category(
                        task_type, category, category_data
                    ))
        
        self._templates_cache[cache_key] = templates
        return templates
    
    def _get_effective_task_type(self, task_type: str, dataset: str = None) -> str:
        """Map dataset names to appropriate task types for template selection"""
        if task_type == "vulnerability_detection" and dataset:
            # Map specific datasets to their corresponding template types
            if dataset == "primevul_pairs" or dataset == "primevul_pair":
                return "vulnerability_detection_pairs"
        return task_type
    
    def _has_language_structure(self, task_config: Dict) -> bool:
        """Check if task has language sub-structure (like unit_test_generation)"""
        # Check if any top-level key contains category data
        for key, value in task_config.items():
            if isinstance(value, dict) and 'system_prompt' in value and 'user_prompt' in value:
                return False  # Direct category structure
            elif isinstance(value, dict):
                # Check if it's a language structure (contains categories)
                sub_values = list(value.values())
                if sub_values and isinstance(sub_values[0], dict) and 'system_prompt' in sub_values[0]:
                    return True  # Language structure
        return False
    
    def _create_templates_from_category(self, task_type: str, category: str, category_data: Dict) -> List[Template]:
        """Create Template objects from category data"""
        templates = []
        
        system_prompt = category_data.get('system_prompt', 'You are a helpful assistant.')
        user_prompts = category_data.get('user_prompt', [])
        
        for prompt_data in user_prompts:
            if isinstance(prompt_data, dict):
                template = Template(
                    task_type=task_type,
                    template_id=prompt_data.get('template_id', 0),
                    template_string=prompt_data.get('template', ''),
                    system_prompt=system_prompt,
                    category=category
                )
                templates.append(template)
        
        return templates
    
    def get_template(self, task_type: str, template_id: int, language: str = None, 
                    category: str = None, dataset: str = None) -> Optional[Template]:
        """Get a specific template by ID"""
        templates = self.load_templates(task_type, language, dataset)
        
        for template in templates:
            if template.template_id == template_id:
                if category is None or template.category == category:
                    return template
        
        return None
    
    def list_available_tasks(self) -> List[str]:
        """List all available tasks in the template data"""
        template_data = self._load_template_data()
        return list(template_data.keys())
    
    def list_available_categories(self, task_type: str, language: str = None) -> List[str]:
        """List all available categories for a task"""
        template_data = self._load_template_data()
        if task_type not in template_data:
            return []
        
        task_config = template_data[task_type]
        
        if self._has_language_structure(task_config):
            if language and language in task_config:
                return list(task_config[language].keys())
            else:
                # Return categories from first language as example
                first_lang = next(iter(task_config.keys()))
                return list(task_config[first_lang].keys())
        else:
            return list(task_config.keys())
    
    def list_available_languages(self, task_type: str) -> List[str]:
        """List all available languages for a task"""
        template_data = self._load_template_data()
        if task_type not in template_data:
            return []
        
        task_config = template_data[task_type]
        
        if self._has_language_structure(task_config):
            return list(task_config.keys())
        else:
            return []  # No language structure