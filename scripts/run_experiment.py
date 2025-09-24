import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pprint
from typing import List, Dict, Optional
from treat.core.base_runner import TaskConfig
def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_specification(config):
    model_spec = config['model_specification']
    print(model_spec.values())
    return model_spec.values()

def load_tasks(config):
    task_list = [(task['task'], task['parameters']) for task in config['tasks'] if task['enabled']]
    return task_list

def create_vulnerability_detection_config(
    models: List[str],
    dataset: Dict[str, str],
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=None,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )

def create_code_generation_config(
    models: List[str],
    dataset: Dict[str, str],
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=language,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )
 
def create_code_translation_config(
    models: List[str],
    dataset: Dict[str, str],
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=language,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )

def create_unit_test_generation_config(
    models: List[str],
    dataset: str,
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=language,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )

def create_code_summarization_config(
    models: List[str],
    dataset: Dict[str, str],
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=language,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )
    
def create_code_review_generation_config(
    models: List[str],
    dataset: Dict[str, str],
    language: str,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        dataset=dataset,
        language=language,  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )

if __name__ == '__main__':
    config = load_yaml("configs/configs.yaml")

    # model_specification
    models, temperature, top_k, top_p = load_model_specification(config)

    # tasks
    tasks = load_tasks(config)
    for task_name, params in tasks:
        if task_name == "unit_test_generation":
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_unit_test_generation_config(
                        models=models,
                        dataset=dataset,  # Dict[str, str]
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.unit_test_generation.runner import TestGenerationRunner
                    ut_runner = TestGenerationRunner(task_config)
                    ut_runner.execute_task()
        if task_name == "code_translation":
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_translation_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_translation.runner import CodeTranslationRunner
                    ct_runner = CodeTranslationRunner(task_config)
                    ct_runner.execute_task()
        if task_name == "code_generation":
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_translation_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_generation.runner import CodeGenerationRunner
                    ct_runner = CodeGenerationRunner(task_config)
                    ct_runner.execute_task()
        if task_name == "vulnerability_detection":
            for dataset in params['datasets']:
                task_config = create_vulnerability_detection_config(
                    models=models,
                    dataset=dataset,
                    language=None,
                    template_categories=params.get('template_categories'),
                    test_samples=params.get('test_samples'),
                    n_requests=params.get('n_requests', 1),
                    parallel_requests=params.get('parallel_requests'),
                    save_dir=params.get('save_dir', "save"),
                    sampling_mode=params.get('sampling_mode'),
                    sample_size=params.get('sample_size'),
                    sampling_seed=params.get('sampling_seed', 42),
                    sampling_manifest_path=params.get('sampling_manifest_path'),
                )

                from treat.runner.vulnerability_detection.runner import VulnerabilityDetectionRunner
                ct_runner = VulnerabilityDetectionRunner(task_config)
                ct_runner.execute_task()
        if task_name == "code_summarization":
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_summarization_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_summarization.runner import CodeSummarizationRunner
                    ct_runner = CodeSummarizationRunner(task_config)
                    ct_runner.execute_task()
        if task_name == "code_review_generation":
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_review_generation_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_review_generation.runner import CodeReviewGenerationRunner
                    ct_runner = CodeReviewGenerationRunner(task_config)
                    ct_runner.execute_task()
        if task_name == 'input_prediction':
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_review_generation_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_reasoning.runner import CodeReasoningRunner
                    ct_runner = CodeReasoningRunner(task_config, 'input_prediction')
                    ct_runner.execute_task()
        if task_name == 'output_prediction':
            for dataset_langauge_map in params['datasets']:
                for dataset, language in dataset_langauge_map.items():
                    task_config = create_code_review_generation_config(
                        models=models,
                        dataset=dataset,
                        language=language,
                        template_categories=params.get('template_categories'),
                        test_samples=params.get('test_samples'),
                        n_requests=params.get('n_requests', 1),
                        parallel_requests=params.get('parallel_requests'),
                        save_dir=params.get('save_dir', "save"),
                        sampling_mode=params.get('sampling_mode'),
                        sample_size=params.get('sample_size'),
                        sampling_seed=params.get('sampling_seed', 42),
                        sampling_manifest_path=params.get('sampling_manifest_path'),
                    )

                    from treat.runner.code_reasoning.runner import CodeReasoningRunner
                    ct_runner = CodeReasoningRunner(task_config, 'output_prediction')
                    ct_runner.execute_task()
# for task, configuration in tasks:
#     if task == 'code_translation':
#         program = CodeTranslation(models, **configuration)
#         program.execute_task()
#     elif task == 'code_generation':
#         program = CodeGeneration(models, **configuration)
#         program.execute_task(parallel_requests=8)
#     elif task == 'code_summarization':
#         program = CodeSummarization(models, **configuration)
#         program.execute_task()
#         # program.evaluate()
#         # program.write_results(have_baseline=True)
#     elif task == 'code_execution':
#         program = CodeExecution(models, **configuration)
#         program.execute_task()
#     elif task == 'input_prediction':
#         configuration['task'] = 'input_prediction'
#         program = CodeExecution(models, **configuration)
#         program.execute_task()
#     elif task == 'output_prediction':
#         configuration['task'] = 'output_prediction'
#         program = CodeExecution(models, **configuration)
#         program.execute_task()
#     elif task == 'vulnerability_detection':
#         # configuration['task'] = 'vulnerability_detection'
#         program = VulnerabilityDetection(models, **configuration)
#         program.execute_task()
#     elif task == 'code_review_generation':
#         configuration['task'] = 'code_review_generation'
#         program = CodeReviewGeneration(models, **configuration)
#         program.execute_task()
#     # program.execute_task()
