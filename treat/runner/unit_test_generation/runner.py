from treat.core.base_runner import BaseTaskRunner, TaskConfig
from typing import Dict, List, Any, Tuple
import traceback
from treat.core.record_manager import RecordManager
from treat.core.template_manager import Template

class TestGenerationRunner(BaseTaskRunner):
    """Unified runner for unit test generation."""
    
    def __init__(self, config: TaskConfig):
        self.task_name = "unit_test_generation"
        super().__init__(config)
    
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

        print(f"🔧 [DEBUG] Processing work item:")
        print(f"   - Model: {model_name}")
        print(f"   - Template ID: {template.template_id}")
        print(f"   - Data ID: {getattr(data, 'id', 'unknown')}")
        print(f"   - Dataset: {getattr(data, 'dataset_name', 'unknown')}")

        try:
            ref_key = RecordManager.make_ref_key(self.config.dataset, model_name, data.idx, self.config.template_categories, template.template_id)
            
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
                "lang": self.config.language,
                "dataset": data.dataset_name,
                "data_idx": data.idx,
                "prompting_category": template.category,
                "prompt_id": prompt_id,
                "prompt_template": template.template_string,
                "wrapped_text": wrapped_text,
                "model_name": model_name,
                "ground_truth": data.ground_truth,
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

        if data.dataset_name == 'primevul_pair':  
            template_info = {
                "code1": data.data1.func,
                "code2": data.data2.func
            }
        elif data.dataset_name == 'primevul':
            template_info = {
                "code": data.func
            }
        
        result = template.get_prompt(model=model, **template_info)
        return result