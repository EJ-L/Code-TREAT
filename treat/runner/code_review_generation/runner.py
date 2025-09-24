from treat.core.base_runner import BaseTaskRunner, TaskConfig
from typing import Dict, List, Any, Optional, Tuple, Iterable, Union
import json, os
from datetime import datetime
import traceback
from treat.core.record_manager import RecordManager
from treat.core.template_manager import Template

class CodeReviewGenerationRunner(BaseTaskRunner):
    """Unified runner for code review generation."""
    
    def __init__(self, config: TaskConfig):
        self.task_name = "code_review_generation"
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
    
    def _save_sampling_manifest(self, items: List[Any]):
        """Save the sampled dataset metadata so the experiment can be replayed."""
        path = self.config.sampling_manifest_path or self._manifest_default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        manifest = {
            "task": self.get_task_name(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_candidates": len(self.dataset),
            "sample_size": len(items),
            "sampling_mode": "random",
            "sampling_seed": self.config.sampling_seed,
            "items": [
                {"dataset": getattr(d, "dataset_name", None), "id": getattr(d, "composite_id", None)}
                for d in items
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[{self.get_task_name()}] Wrote sampling manifest to: {path}")


    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        data, model, template = work_item
        model_name = getattr(model, "model_name", "unknown_model")

        try:
            ref_key = RecordManager.make_ref_key(self.config.dataset, model_name, data.owner + '/' + data.repo, data.pr_id, data.diff_hunk, self.config.template_categories, template.template_id)
            
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
                "dataset": self.config.dataset,
                "repo": data.owner + "/" + data.repo,
                "diff_hunk": data.diff_hunk,
                "pr_id": data.pr_id,
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
            'diff_hunk': data.diff_hunk,
        }
        
        result = template.get_prompt(model=model, **template_info)
        return result