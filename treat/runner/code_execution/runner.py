from typing import Dict, List, Any, Tuple
from treat.core.base_runner import BaseTaskRunner, TaskConfig
import traceback
from treat.core.record_manager import RecordManager
from treat.core.template_manager import Template


class CodeExecutionRunner(BaseTaskRunner):
    """Unified runner for code generation."""
    def __init__(self, config: TaskConfig, task: str):
        if task not in ['input_prediction', 'output_prediction']:
            raise ValueError(f"task {task} not supported for code execution")
        self.task_name = task
        super().__init__(config)
        self.lang = self.config.language
        
    def get_task_name(self):
        return self.task_name
        
    def create_work_items(self, dataset: List[Any], templates: List[Any], models: List[Any]) -> List[Tuple]:
        """Create work items for parallel execution"""
        work_items = []
        for data in dataset:
            for model in models:
                for template in templates:
                    work_items.append((data, model, template))
                        
        return work_items
        
    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        """Process a single work item"""
        data, model, template = work_item
        model_name = getattr(model, "model_name", "unknown_model")
        
        try:
            ref_key = RecordManager.make_ref_key(self.config.dataset, model_name, data.id, data.test_case_idx, self.lang, self.config.template_categories, template.template_id)
        
            if ref_key is None:
                return {"status": "skipped", "reason": "no_prompt_id"}

            if self.record_manager.is_ref_processed(ref_key):
                return {"status": "skipped", "reason": "already_processed"}
            # Check if already processed
                        
            # Build prompt
            message, wrapped_text = self._build_prompt(data, model, template)
            
            # Get responses with parallel fetching
            response_list = self.fetch_responses_parallel(
                model, message, self.config.n_requests
            )
            
            prompt_id = getattr(template, "template_id", None)
            
            # Build result record (unified format)
            result = {
                "task": self.task_name,
                "ref_key": ref_key,
                "dataset": data.dataset,
                "id": data.id,
                "lang": self.lang,
                "difficulty": data.difficulty,
                "code/function": data.function,
                "prompt_category": template.category,
                "prompt_id": prompt_id,
                "prompt_template": template.template_string,
                "wrapped_text": wrapped_text,
                "model_name": model_name,
                "response": response_list,
            }
            
            # Save result
            self.record_manager.save_record(model_name, result)            
            return {"status": "completed"}
            
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
            
    def _build_prompt(
        self, 
        data: Any, 
        model: Any, 
        template: Any
    ) -> Tuple[Any, str]:
        """Build unified prompt for code generation"""

        if self.task_name == 'input_prediction':        # Build template info (unified)
            template_info = {
                'function': data.function,
                'assertion_query': data.input_prediction_query,
            }
        elif self.task_name == 'output_prediction':
            template_info = {
                'function': data.function,
                'assertion_query': data.output_prediction_query,
            }
        
        result = template.get_prompt(model=model, **template_info)
        return result


