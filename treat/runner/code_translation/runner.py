from treat.core.base_runner import BaseTaskRunner, TaskConfig
from typing import Dict, List, Any, Tuple
import traceback
from treat.core.record_manager import RecordManager
from treat.core.template_manager import Template

class CodeTranslationRunner(BaseTaskRunner):
    """Unified runner for code translation."""
    
    def __init__(self, config: TaskConfig):
        self.task_name = "code_translation"
        super().__init__(config)
        self.source_lang, self.target_lang = self.config.language.split("->")
    def get_task_name(self):
        return self.task_name
    
    def create_work_items(self, dataset: List[Any], templates: List[Any], models: List[Any]) -> List[Tuple]:
        work_items: List[Tuple] = []
        for data in dataset:
            for model in models:
                for template in templates:
                    work_items.append((data, model, template))
        return work_items
    
    
    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        data, model, template = work_item
        model_name = getattr(model, "model_name", "unknown_model")
        
        try:
            ref_key = RecordManager.make_ref_key(self.config.dataset, model_name, data.id, f"{self.source_lang}->{self.target_lang}", self.config.template_categories, template.template_id)
            
            if ref_key is None:
                return {"status": "skipped", "reason": "no_prompt_id"}

            if self.record_manager.is_ref_processed(ref_key):
                return {"status": "skipped", "reason": "already_processed"}

            # Build prompt
            message, wrapped_text = self._build_prompt(data, model, template)
            
            # Fetch responses
            response_list = self.fetch_responses_parallel(
                model, message, self.config.n_requests
            )

            prompt_id = getattr(template, "template_id", None)

            result = {
                "task": self.task_name,
                "ref_key": ref_key,
                "dataset": self.config.dataset,
                "difficulty": data.difficulty,
                "id": data.id,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "modality": self.source_lang + "->" + self.target_lang,
                "prompting_category": template.category,
                "prompt_id": prompt_id,
                "prompt_template": template.template_string,
                "wrapped_text": wrapped_text,
                "model_name": model_name,
                "response": response_list,
            }

            self.record_manager.save_record(model_name, result)
            return {"status": "completed"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
        
    def _build_prompt(
        self,
        data: Any,
        model: Any,
        template: Template
    ) -> Tuple[Any, str]:        
        template_info = {
            "SL": self.source_lang,
            "TL": self.target_lang,
            "SC": data.source_code,
        }
        
        result = template.get_prompt(model=model, **template_info)
        return result