from polyeval.parsing import parse
from polyeval.eval import ProjectTemplate, EvalStatus, gen_codes, gen_codes_for_single_file, create_project

from tqdm import tqdm
import sys
import os
import argparse
import json

from pebble import ProcessPool

from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

langs = ["cpp", "csharp", "dart", "go", "java", "javascript", "kotlin", 
           "php", "python", "ruby", "rust", "scala", "swift", "typescript"]

suffix = {
    "cpp": "cpp",
    "csharp": "cs",
    "dart": "dart",
    "go": "go",
    "java": "java",
    "javascript": "js",
    "kotlin": "kt",
    "php": "php",
    "python": "py",
    "ruby": "rb",
    "rust": "rs",
    "scala": "scala",
    "swift": "swift",
    "typescript": "ts"
}

cur_langs = ['python', 'java']

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--output", type=str, default="./output_data/result.json")

args = parser.parse_args()
input_file = args.input
output_file = args.output



# print(f"Loading problem description and solution...")
# with open(os.path.join(ROOT, "./data/poly_humanevalv4.testdsl"), "r") as f:
#     desc_str = f.readlines()
#     problem_start = []
#     for idx, line in enumerate(desc_str):
#         if 'problem HumanEval/' in line:
#             problem_start.append(idx)
#     problems = []
#     for idx, line_index in enumerate(problem_start):
#         if idx == len(problem_start) - 1:
#             problems.append(''.join(desc_str[line_index:]))
#         else:
#             problems.append(''.join(desc_str[line_index:problem_start[idx+1]]))
#     print(len(problems))
#     for idx, p in enumerate(problems):
#         # if idx < 38:
#         #     continue
#         print("in problem:", idx)
#         problems = parse(p)
input_file = "./data/poly_humaneval_sol.json"
with open(os.path.join(ROOT, input_file), "r") as f:
    solutions = json.load(f)
print("Solutions")
print(solutions['python'].keys())
print(f"Loading project templates...")
templates = {}
fallback_templates = {}
for lang in tqdm(cur_langs):
    print(f"Loading {lang} ...")
    templates[lang] = ProjectTemplate(os.path.join(ROOT, "./project-templates/default", lang))
    if os.path.exists(os.path.join(ROOT, "./project-templates/with-dependencies", lang)):
        print(f"Loading {lang} with dependencies ...")
        fallback_templates[lang] = ProjectTemplate(os.path.join(ROOT, "./project-templates/with-dependencies", lang))
    else:
        fallback_templates[lang] = None


def evaluate(lang, problem, solution, template, fallback_template):
    p_name = problem.name.replace("/", "_") + "_" + lang
    codes = gen_codes_for_single_file(lang=lang, problem=problem, target_code=solution)
    proj = create_project(template, f"{lang}-{p_name}-sf", codes,
                                      root=os.path.join(ROOT, ".polyeval/"), overwrite=True)
    ret_stat, msg = proj.evaluate(compile_timeout=10, run_timeout=10)
    if ret_stat == EvalStatus.Pass:
        return True
    if fallback_template is not None:
        codes = gen_codes(lang=lang, problem=problem, target_code=solution)
        proj = create_project(fallback_template, f"{lang}-{p_name}", codes, root=os.path.join(ROOT, ".polyeval/"), overwrite=True)
        ret_stat, msg = proj.evaluate(compile_timeout=30, run_timeout=10)
        if ret_stat == EvalStatus.Pass:
            return True
    return False

if __name__ == "__main__":
    results = {}

    with ProcessPool(max_workers=10) as pool:
        futures = []
        for src_lang in solutions:
            if src_lang not in cur_langs:
                continue
            for tgt_lang in solutions[src_lang]:
                print(f"Evaluating {src_lang} -> {tgt_lang}")
                if tgt_lang not in cur_langs:
                    continue
                assert len(solutions[src_lang][tgt_lang]) == 164
                for i in range(0, 164):
                    if i != 14:
                        solution = solutions[src_lang][tgt_lang][i]
                        problem = list(problems.values())[i]
                        name = f"{src_lang}-{tgt_lang}-{i}"
                        future = pool.schedule(evaluate, args=[tgt_lang, problem, solution, templates[tgt_lang], fallback_templates[tgt_lang]], timeout=150)
                        futures.append([src_lang, tgt_lang, i, future])
        
        for src_lang, tgt_lang, i, future in tqdm(futures):
            try:
                ret = future.result()
                if src_lang not in results:
                    results[src_lang] = {}
                if tgt_lang not in results[src_lang]:
                    results[src_lang][tgt_lang] = [None for _ in range(164)]
                results[src_lang][tgt_lang][i] = ret
            except Exception as e:
                print(e)
                results[src_lang][tgt_lang][i] = False


    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)