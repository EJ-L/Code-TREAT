from typing import Dict, List, Any, Optional, Tuple, Iterable, Union

class DatasetManager:
    def __init__(self, task_name: str, dataset: str, language: str):
        self.task_name = task_name
        self.dataset = dataset
        self.language = language
        
        
    def load_dataset(self) -> List[Any]:
        if self.task_name == "code_generation":
            from treat.runner.code_generation.dataloader import DataLoader as CodeGenerationDataLoader
            return CodeGenerationDataLoader(self.dataset, self.language).load_data()
        if self.task_name == "code_translation":
            from treat.runner.code_translation.dataloader import DataLoader as CodeTranslationDataLoader
            return CodeTranslationDataLoader(self.dataset, self.language).load_data()
        if self.task_name == "vulnerability_detection":
            from treat.runner.vulnerability_detection.dataloader import DataLoader as VulnerabilityDetectionDataLoader
            return VulnerabilityDetectionDataLoader(self.dataset).load_data()
        if self.task_name == "code_summarization":
            from treat.runner.code_summarization.dataloader import DataLoader as CodeSummarizationDataLoader
            return CodeSummarizationDataLoader(self.dataset, self.language).load_data()
        if self.task_name == "code_review_generation":
            from treat.runner.code_review_generation.dataloader import DataLoader as CodeReviewGenerationDataLoader
            return CodeReviewGenerationDataLoader(self.dataset, self.language).load_data()
        if self.task_name == "unit_test_generation":
            # Lazy import to avoid circular dependency
            from treat.runner.unit_test_generation.dataloader import DataLoader as UnitTestGenerationDataLoader
            return UnitTestGenerationDataLoader(self.dataset).load_data()
        if self.task_name == "input_prediction":
            from treat.runner.code_reasoning.dataloader import DataLoader as InputPredictionDataLoader
            return InputPredictionDataLoader(self.dataset, self.language).load_data()
        if self.task_name == "output_prediction":
            from treat.runner.code_reasoning.dataloader import DataLoader as OutputPredictionDataLoader
            return OutputPredictionDataLoader(self.dataset, self.language).load_data()
        
        raise ValueError(f"Unknown task name: {self.task_name}")