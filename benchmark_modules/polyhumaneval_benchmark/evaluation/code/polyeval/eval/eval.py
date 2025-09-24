import os
import shutil

from typing import Dict

from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.eval.project import Project
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.eval.project_template import ProjectTemplate
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.objects.problem import Problem
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.generators.lang_generator import create_generator


def gen_codes(lang: str, problem: Problem, target_code: str):
    generator = create_generator(lang, problem)
    codes = generator.gen_codes()
    codes["target"] = target_code
    return codes


def gen_codes_for_single_file(lang: str, problem: Problem, target_code: str):
    generator = create_generator(lang, problem)
    codes = generator.gen_codes()
    main_code = target_code + "\n\n" + codes["main"]
    return {"main": main_code}


def create_project(template: ProjectTemplate, name: str, codes: Dict[str, str], root: str = "./.polyeval",
                   overwrite: bool = False):
    root = os.path.join(root)
    target_dir_name = os.path.join(root, name)
    if overwrite and os.path.exists(target_dir_name):
        shutil.rmtree(target_dir_name)
    project = Project(target_dir_name, template)
    project.set_codes(codes)
    return project

# Further improved version
def create_project(template: ProjectTemplate, name: str, codes: Dict[str, str], root: str = "./.polyeval",
                   overwrite: bool = False, imports: str = ""):
    root = os.path.join(root)
    target_dir_name = os.path.join(root, name)
    if overwrite and os.path.exists(target_dir_name):
        shutil.rmtree(target_dir_name)
    project = Project(target_dir_name, template)
    project.set_codes(codes, imports)
    return project

def single_evaluation(lang: str, problem: Problem, target_code: str,
                      template: ProjectTemplate,
                      name: str = "main", root: str = "./.polyeval",
                      timeout_compile: int = 60,
                      timeout_run: int = 60,
                      keep_after_eval: bool = False,
                      keep_when_fail: bool = False):
    codes = gen_codes_for_single_file(lang, problem, target_code)
    project = create_project(template, name, codes, root, overwrite=True)
    return project.evaluate(timeout_compile, timeout_run, keep_after_eval, keep_when_fail)