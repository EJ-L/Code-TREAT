from __future__ import annotations
import importlib
from typing import Type
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.generators.base import LangGeneratorBase
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.generators.python import generator_class as python_generator_class
from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.objects.problem import Problem
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
def get_generator_class(lang: str) -> Type[LangGeneratorBase]:
    if lang == "python":
        return python_generator_class
    plugin_package_name = f"benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.plugins.{lang}"
    plugin_package = importlib.import_module(plugin_package_name)
    if hasattr(plugin_package, "generator_class"):
        return plugin_package.generator_class
    else:
        raise ValueError(f"No generator for {lang} found")


def create_generator(lang: str, problem: Problem) -> LangGeneratorBase:
    return get_generator_class(lang)(problem)
get_generator_class("java")