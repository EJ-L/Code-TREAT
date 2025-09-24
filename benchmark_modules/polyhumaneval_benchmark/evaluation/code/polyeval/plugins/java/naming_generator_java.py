from __future__ import annotations

from benchmark_modules.polyhumaneval_benchmark.evaluation.code.polyeval.generators.base import *


class NamingGeneratorJava(NamingGeneratorBase):
    global_func_naming = "camelCase"
    global_var_naming = "camelCase"
    member_func_naming = "camelCase"
    member_var_naming = "camelCase"
    arg_naming = "camelCase"
    temp_var_naming = "camelCase"
