#!/usr/bin/env python3
"""Generate the overall LaTeX table combining metrics from all tasks.

This script aggregates metrics from different evaluation tasks:
- CG: Code Generation (average pass@1 from GeeksforGeeks and HackerRank)
- CS: Code Summarization (LLMJudge scores normalized to 0-100)
- CT: Code Translation (average pass@1 from PolyHumanEval and HackerRank)  
- CR: Code Repair (average pass@1 from input_prediction and output_prediction)
- CRv: Code Review (average scores normalized by dividing by 5, then *100)
- TG: Test Generation (average line coverage from unit_test_generation)
- VD: Vulnerability Detection (accuracy from vulnerability_detection)

Usage:
    python construct_tables/overall_table.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any


# Model display order (same as table_updater.py)
MODEL_DISPLAY_ORDER = [
    "Gemini-2.5-Pro",
    "GPT-4.1",
    "Claude-3.7-Sonnet", 
    "o3-mini (Med)",
    "o4-mini (Med)",
    "GPT-5",
    "DeepSeek-R1 (0528)",
    "DeepSeek-R1",
    "Claude-Sonnet-4",
    "Claude-3.5-Sonnet",
    "DeepSeek-V3",
    "GPT-4o",
    "Grok-3-Mini (High)",
    "Qwen3-235B-A22B",
    "Qwen3-32B",
    "Qwen3-30B-A3B", 
    "Qwen2.5-72B-Instruct",
    "GPT-4-turbo",
    "Claude-3.5-Haiku",
    "Qwen2.5-Coder-32B-Instruct",
    "Gemma-3-27B-Instruct",
    "LLaMA-4-Scout-17B-16E-Instruct",
    "LLaMA-3.3-70B-Instruct",
    "GPT-3.5-turbo",
    "LLaMA-3.1-70B-Instruct",
    "LLaMA-3.1-8B-Instruct",
]

# Model name mappings
MODEL_ALIASES = {
    "anthropic_claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "anthropic_claude-sonnet-4": "Claude-Sonnet-4",
    "claude-3-5-haiku-20241022": "Claude-3.5-Haiku", 
    "claude-3-5-sonnet-20241022": "Claude-3.5-Sonnet",
    "deepseek-ai_DeepSeek-R1": "DeepSeek-R1",
    "deepseek_deepseek-r1-0528": "DeepSeek-R1 (0528)",
    "deepseek-chat": "DeepSeek-V3",
    "DeepSeek-V3 (0324)": "DeepSeek-V3",
    "grok-3-mini-beta": "Grok-3-Mini (High)",
    "google_gemini-2.5-pro-preview-05-06": "Gemini-2.5-Pro-05-06",
    "google_gemma-3-27b-it": "Gemma-3-27B-Instruct",
    "Gemini-2.5-Pro-05-06": "Gemini-2.5-Pro-05-06",
    "meta-llama_Llama-3.3-70B-Instruct": "LLaMA-3.3-70B-Instruct",
    "meta-llama_Llama-4-Scout-17B-16E-Instruct": "LLaMA-4-Scout-17B-16E-Instruct",
    "meta-llama_Meta-Llama-3.1-70B-Instruct": "LLaMA-3.1-70B-Instruct",
    "meta-llama_Meta-Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "Qwen_Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct", 
    "Qwen_Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "Qwen_Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen2-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Llama-3.1-70B-Instruct": "LLaMA-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct": "LLaMA-3.3-70B-Instruct", 
    "Llama-4-Scout-17B-16E-Instruct": "LLaMA-4-Scout-17B-16E-Instruct",
    "llama-3.1-70b-instruct": "LLaMA-3.1-70B-Instruct",
    "llama-3.1-8b-instruct": "LLaMA-3.1-8B-Instruct",
    "o3-mini": "o3-mini (Med)",
    "o4-mini": "o4-mini (Med)",
    # Mappings for Gemini model names
    "google_gemini-2.5-pro-preview-05-06": "Gemini-2.5-Pro",
    "Gemini-2.5-Pro-Preview-05-06": "Gemini-2.5-Pro",
    "Gemini-2.5-Pro-05-06": "Gemini-2.5-Pro",
    # Reverse mappings - full names to short names
    "Claude-3.5-Sonnet-20241022": "Claude-3.5-Sonnet",
    "Claude-3.5-Haiku-20241022": "Claude-3.5-Haiku",
    "GPT-4o-2024-11-20": "GPT-4o",
    "GPT-4-turbo-2024-04-09": "GPT-4-turbo",
    "GPT-3.5-turbo-0125": "GPT-3.5-turbo",
    "GPT-4.1-2025-04-14": "GPT-4.1"
}


def normalize_model_name(raw_name: str) -> str:
    """Normalize model name using aliases mapping."""
    return MODEL_ALIASES.get(raw_name, raw_name)


def read_jsonl(file_path: Path) -> List[Dict]:
    """Read a JSONL file and return list of records."""
    records = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def collect_pass_at_1_metrics(results_dir: Path, task: str, datasets: List[str]) -> Dict[str, float]:
    """Collect pass@1 metrics for a task across specified datasets."""
    
    task_dir = results_dir / task
    if not task_dir.exists():
        return {}
    
    model_scores = defaultdict(list)
    
    for dataset in datasets:
        dataset_dir = task_dir / dataset / "evaluations"
        if not dataset_dir.exists():
            continue
            
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            model_name = normalize_model_name(jsonl_file.stem)
            records = read_jsonl(jsonl_file)
            
            for record in records:
                metrics = record.get("metrics", {})
                if isinstance(metrics, dict):
                    pass_at_1 = metrics.get("pass@1")
                    if isinstance(pass_at_1, (int, float)):
                        model_scores[model_name].append(float(pass_at_1))
    
    # Calculate averages
    model_averages = {}
    for model, scores in model_scores.items():
        if scores:
            model_averages[model] = sum(scores) / len(scores)
    
    return model_averages


def collect_code_summarization_metrics(results_dir: Path) -> Dict[str, float]:
    """Collect LLMJudge metrics from code summarization, normalized to 0-100."""
    
    task_dir = results_dir / "code_summarization"
    if not task_dir.exists():
        return {}
    
    model_scores = defaultdict(list)
    
    for dataset_dir in task_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        eval_dir = dataset_dir / "evaluations"
        if not eval_dir.exists():
            continue
            
        for jsonl_file in eval_dir.glob("*.jsonl"):
            model_name = normalize_model_name(jsonl_file.stem)
            records = read_jsonl(jsonl_file)
            
            for record in records:
                metrics = record.get("metrics", {})
                if isinstance(metrics, dict):
                    llm_judge = metrics.get("LLMJudge")
                    if isinstance(llm_judge, dict):
                        # Extract all numeric values from LLMJudge metrics
                        values = []
                        for key, value in llm_judge.items():
                            if isinstance(value, (int, float)):
                                values.append(float(value))
                            elif isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, (int, float)):
                                        values.append(float(subvalue))
                        
                        if values:
                            avg_score = sum(values) / len(values)
                            # Normalize from 0-5 to 0-100 (same as CRv)
                            model_scores[model_name].append((avg_score / 5.0) * 100)
    
    # Calculate averages
    model_averages = {}
    for model, scores in model_scores.items():
        if scores:
            model_averages[model] = sum(scores) / len(scores)
    
    return model_averages


def collect_code_review_metrics(results_dir: Path) -> Dict[str, float]:
    """Collect code review metrics, normalized to 0-100."""
    
    task_dir = results_dir / "code_review_generation"
    if not task_dir.exists():
        return {}
    
    model_scores = defaultdict(list)
    
    for dataset_dir in task_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        eval_dir = dataset_dir / "evaluations"
        if not eval_dir.exists():
            continue
            
        for jsonl_file in eval_dir.glob("*.jsonl"):
            model_name = normalize_model_name(jsonl_file.stem)
            records = read_jsonl(jsonl_file)
            
            for record in records:
                metrics = record.get("metrics", {})
                if isinstance(metrics, dict):
                    values = []
                    # Extract scores from each evaluator
                    for evaluator_name, evaluator_scores in metrics.items():
                        if isinstance(evaluator_scores, list):
                            # evaluator_scores is a list of numeric ratings
                            for score in evaluator_scores:
                                if isinstance(score, (int, float)):
                                    values.append(float(score))
                    
                    if values:
                        avg_score = sum(values) / len(values)
                        # Normalize from 0-5 to 0-100
                        model_scores[model_name].append((avg_score / 5.0) * 100)
    
    # Calculate averages
    model_averages = {}
    for model, scores in model_scores.items():
        if scores:
            model_averages[model] = sum(scores) / len(scores)
    
    return model_averages


def collect_test_generation_metrics(results_dir: Path) -> Dict[str, float]:
    """Collect line coverage metrics from unit test generation."""
    
    symprompt_dir = results_dir / "unit_test_generation" / "symprompt" / "evaluations"
    if not symprompt_dir.exists():
        return {}
    
    model_scores = defaultdict(list)
    
    for jsonl_file in symprompt_dir.glob("*.jsonl"):
        model_name = normalize_model_name(jsonl_file.stem)
        records = read_jsonl(jsonl_file)
        
        for record in records:
            metrics = record.get("metrics", [])
            if metrics and isinstance(metrics, list) and len(metrics) > 0:
                first_metric = metrics[0]
                if isinstance(first_metric, dict):
                    line_coverage = first_metric.get("line_coverage")
                    if isinstance(line_coverage, (int, float)):
                        model_scores[model_name].append(float(line_coverage))
    
    # Calculate averages
    model_averages = {}
    for model, scores in model_scores.items():
        if scores:
            model_averages[model] = sum(scores) / len(scores)
    
    return model_averages


from pathlib import Path
from typing import Dict
from collections import defaultdict

def collect_code_generation_metrics(results_dir: Path) -> Dict[str, float]:
    """
    Compute overall pass@1 (%) per model by pooling ALL samples across:
      datasets = {geeksforgeeks, hackerrank}
      languages = {python, java, ...}
    Definition: overall = (#passes across all samples) / (total #samples) * 100.
    """
    task_dir = results_dir / "code_generation"
    if not task_dir.exists():
        return {}

    # per-model accumulators
    total_pass_sum = defaultdict(float)   # sum of pass@1 (0 or 1) across all samples
    total_count    = defaultdict(int)     # number of samples

    for dataset in ("geeksforgeeks", "hackerrank"):
        dataset_dir = task_dir / dataset / "evaluations"
        if not dataset_dir.exists():
            continue

        for jsonl_file in dataset_dir.glob("*.jsonl"):
            if "backup" in jsonl_file.name.lower():
                continue

            model_name = normalize_model_name(jsonl_file.stem)
            for rec in read_jsonl(jsonl_file):
                metrics = rec.get("metrics") or {}
                v = metrics.get("pass@1")
                if v is None:
                    continue
                try:
                    v = float(v)
                except Exception:
                    continue
                # normalize if someone stored percentages
                if v > 1.0:
                    v = v / 100.0
                # clamp to [0,1]
                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0

                total_pass_sum[model_name] += v
                total_count[model_name]    += 1

    # final overall (%) per model
    out = {}
    for m in total_pass_sum:
        if total_count[m] > 0:
            out[m] = (total_pass_sum[m] / total_count[m]) * 100.0
    return out

def collect_vulnerability_detection_metrics(results_dir: Path) -> Dict[str, float]:
    """Collect accuracy metrics from vulnerability detection."""
    
    # VD scores are in vulnerability_detection/primevul/evaluations/vul_detect_score.json
    vd_score_file = results_dir / "vulnerability_detection" / "primevul" / "evaluations" / "vul_detect_score.json"
    
    if not vd_score_file.exists():
        return {}
    
    model_scores = {}
    
    try:
        data = json.loads(vd_score_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for raw_model, model_data in data.items():
                model = normalize_model_name(str(raw_model))
                if isinstance(model_data, dict):
                    primevul_metrics = model_data.get("primevul")
                    if isinstance(primevul_metrics, dict):
                        accuracy = primevul_metrics.get("accuracy")
                        if isinstance(accuracy, (int, float)):
                            model_scores[model] = float(accuracy) * 100
    except json.JSONDecodeError:
        pass
    
    return model_scores


def calculate_average_ranks(metrics_by_model: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, float]:
    """Calculate average rank for each model across available metrics."""
    
    task_names = ["CG", "CS", "CT", "CR", "CRv", "TG", "VD"]
    model_ranks = defaultdict(list)
    
    for task in task_names:
        # Get models with valid scores for this task
        valid_scores = {
            model: scores[task] 
            for model, scores in metrics_by_model.items()
            if scores[task] is not None
        }
        
        if len(valid_scores) < 2:
            continue
            
        # Sort by score (descending) to get rankings
        sorted_models = sorted(valid_scores.items(), key=lambda x: (-x[1], x[0]))
        
        # Assign ranks (handle ties with average ranking)
        current_rank = 1
        i = 0
        while i < len(sorted_models):
            # Find all models with the same score (ties)
            j = i
            current_score = sorted_models[i][1]
            while j < len(sorted_models) and abs(sorted_models[j][1] - current_score) < 1e-9:
                j += 1
            
            # Calculate average rank for tied models
            if j - i == 1:
                # No tie
                rank = current_rank
            else:
                # Tie: average the ranks
                rank = (current_rank + current_rank + (j - i) - 1) / 2.0
            
            # Assign this rank to all tied models
            for k in range(i, j):
                model_name = sorted_models[k][0]
                model_ranks[model_name].append(rank)
            
            current_rank += (j - i)
            i = j
    
    # Calculate average rank across all tasks for each model
    average_ranks = {}
    for model, ranks in model_ranks.items():
        if ranks:
            average_ranks[model] = sum(ranks) / len(ranks)
    
    return average_ranks


def generate_latex_table(metrics_by_model: Dict[str, Dict[str, Optional[float]]], 
                        average_ranks: Dict[str, float], precision: int = 1) -> str:
    """Generate the overall LaTeX table."""
    
    # Sort models by average rank (lower is better)
    sorted_models = sorted(
        [(model, average_ranks.get(model, 999)) for model in MODEL_DISPLAY_ORDER 
         if model in metrics_by_model],
        key=lambda x: x[1]
    )
    
    # Normalize rankings to 1-N range
    normalized_ranks = {}
    for i, (model, _) in enumerate(sorted_models):
        normalized_ranks[model] = i + 1
    
    # Get top 3 in each task for highlighting
    task_names = ["CG", "CS", "CT", "CR", "CRv", "TG", "VD"]
    top_performers = {}
    
    for task in task_names:
        valid_scores = [
            (model, scores[task]) 
            for model, scores in metrics_by_model.items()
            if scores[task] is not None
        ]
        valid_scores.sort(key=lambda x: -x[1])
        
        top_performers[task] = {
            valid_scores[i][0]: i+1 
            for i in range(min(3, len(valid_scores)))
        }
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Overall model performance (\\%) on general coding tasks. The top three performing results on each task are highlighted in \\colorbox{firstbg}{\\textcolor{firsttext}{green (\\small $1^{st}$)}}, \\colorbox{secondbg}{\\textcolor{secondtext}{orange (\\small $2^{nd}$)}}, and \\colorbox{thirdbg}{\\textcolor{thirdtext}{blue (\\small $3^{rd}$)}} backgrounds, respectively.}")
    lines.append("\\label{tab:model_performance}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{l|ccccccc|c}")
    lines.append("\\toprule")
    lines.append("\\rowcolor{gray!20}  \\diagbox{\\textbf{Model Name}}{\\textbf{Tasks}} & \\bf CG & \\textbf{CS} & \\textbf{CT} & \\textbf{CR} & \\textbf{CRv} & \\textbf{TG} & \\textbf{VD} & \\textbf{Avg. Rank} \\\\")
    lines.append("\\midrule")
    
    for model, _ in sorted_models:
        scores = metrics_by_model[model]
        
        # Format each score with highlighting
        formatted_scores = []
        for task in task_names:
            score = scores[task]
            if score is None:
                formatted_scores.append("--")
            else:
                score_str = f"{score:.{precision}f}"
                
                # Check if this model is in top 3 for this task
                rank = top_performers[task].get(model)
                if rank == 1:
                    formatted_scores.append(f"\\cellcolor{{firstbg}}\\textcolor{{firsttext}}{{\\textbf{{{score_str}}}}}")
                elif rank == 2:
                    formatted_scores.append(f"\\cellcolor{{secondbg}}\\textcolor{{secondtext}}{{{score_str}}}")
                elif rank == 3:
                    formatted_scores.append(f"\\cellcolor{{thirdbg}}\\textcolor{{thirdtext}}{{{score_str}}}")
                else:
                    formatted_scores.append(score_str)
        
        # Use normalized rank (1 to N)
        normalized_rank = normalized_ranks[model]
        
        # Create table row
        row = f"{model} & {' & '.join(formatted_scores)} & {normalized_rank} \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate overall performance LaTeX table")
    parser.add_argument("--results-dir", type=Path,
                       default=Path(__file__).parent.parent / "results",
                       help="Root directory containing evaluation results")
    parser.add_argument("--output", type=Path,
                       default=Path(__file__).parent.parent / "latex_table" / "overall.tex",
                       help="Output LaTeX file path")
    parser.add_argument("--precision", type=int, default=1,
                       help="Decimal places for percentages")
    
    args = parser.parse_args()
    
    print("Collecting metrics from all tasks...")
    
    # Collect metrics from all tasks
    metrics_by_model = defaultdict(lambda: {
        "CG": None, "CS": None, "CT": None, "CR": None, 
        "CRv": None, "TG": None, "VD": None
    })
    
    # CG: Code Generation (average of Python Overall and Java Overall)
    cg_metrics = collect_code_generation_metrics(args.results_dir)
    for model, score in cg_metrics.items():
        metrics_by_model[model]["CG"] = score
    
    # CS: Code Summarization 
    cs_metrics = collect_code_summarization_metrics(args.results_dir)
    for model, score in cs_metrics.items():
        metrics_by_model[model]["CS"] = score
    
    # CT: Code Translation (PolyHumanEval + HackerRank)
    ct_metrics = collect_pass_at_1_metrics(args.results_dir, "code_translation",
                                          ["polyhumaneval", "hackerrank"])
    for model, score in ct_metrics.items():
        metrics_by_model[model]["CT"] = score * 100
    
    # CR: Code Repair (input_prediction + output_prediction)
    input_metrics = collect_pass_at_1_metrics(args.results_dir, "input_prediction",
                                             ["geeksforgeeks", "hackerrank"])
    output_metrics = collect_pass_at_1_metrics(args.results_dir, "output_prediction", 
                                              ["geeksforgeeks", "hackerrank"])
    
    all_models = set(input_metrics.keys()) | set(output_metrics.keys())
    for model in all_models:
        scores = []
        if model in input_metrics:
            scores.append(input_metrics[model])
        if model in output_metrics:
            scores.append(output_metrics[model])
        if scores:
            metrics_by_model[model]["CR"] = (sum(scores) / len(scores)) * 100
    
    # CRv: Code Review
    crv_metrics = collect_code_review_metrics(args.results_dir)
    for model, score in crv_metrics.items():
        metrics_by_model[model]["CRv"] = score
    
    # TG: Test Generation (line coverage)
    tg_metrics = collect_test_generation_metrics(args.results_dir)
    for model, score in tg_metrics.items():
        metrics_by_model[model]["TG"] = score
    
    # VD: Vulnerability Detection
    vd_metrics = collect_vulnerability_detection_metrics(args.results_dir)
    for model, score in vd_metrics.items():
        metrics_by_model[model]["VD"] = score
    
    # Calculate average ranks
    average_ranks = calculate_average_ranks(metrics_by_model)
    
    # Generate LaTeX table
    latex_content = generate_latex_table(metrics_by_model, average_ranks, args.precision)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_content + "\n", encoding="utf-8")
    
    print(f"Overall table written to {args.output}")
    print(f"Found metrics for {len(metrics_by_model)} models")
    
    # Print summary
    for task in ["CG", "CS", "CT", "CR", "CRv", "TG", "VD"]:
        count = sum(1 for scores in metrics_by_model.values() if scores[task] is not None)
        print(f"  {task}: {count} models")


if __name__ == "__main__":
    main()