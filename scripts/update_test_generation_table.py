#!/usr/bin/env python3
"""Generate a LaTeX table for unit test generation results from symprompt evaluation data.

This script processes the unit test generation evaluation results and creates a
LaTeX table showing model performance on the Symprompt-Python dataset.

The table shows three metrics for each model:
- CSR (Correctness Score Ratio): Proportion of tests that are correct (0-1)
- Cov_L (Line Coverage): Percentage of lines covered by the tests (0-100)
- Cov_Br (Branch Coverage): Percentage of branches covered by the tests (0-100)

Entries where branch_coverage is -1 are skipped as requested.

Usage:
    python construct_tables/unit_test_generation_table.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Model display order from the main table_updater.py
MODEL_DISPLAY_ORDER = [
    "Gemini-2.5-Pro-Preview-05-06",
    "GPT-4.1-2025-04-14", 
    "Claude-3.7-Sonnet",
    "o3-mini (Med)",
    "o4-mini (Med)",
    "GPT-5",
    "DeepSeek-R1 (0528)",
    "DeepSeek-R1",
    "Claude-Sonnet-4",
    "Claude-3.5-Sonnet-20241022",
    "DeepSeek-V3",
    "GPT-4o-2024-11-20",
    "Grok-3-Mini (High)",
    "Qwen3-235B-A22B",
    "Qwen3-32B",
    "Qwen3-30B-A3B",
    "Qwen2.5-72B-Instruct",
    "GPT-4-turbo-2024-04-09",
    "Claude-3.5-Haiku-20241022",
    "Qwen2.5-Coder-32B-Instruct",
    "Gemma-3-27B-Instruct",
    "LLaMA-4-Scout-17B-16E-Instruct",
    "LLaMA-3.3-70B-Instruct",
    "GPT-3.5-turbo-0125",
    "LLaMA-3.1-70B-Instruct",
    "LLaMA-3.1-8B-Instruct",
]

# Model name mappings from table_updater.py
MODEL_ALIASES = {
    "anthropic_claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "anthropic_claude-sonnet-4": "Claude-Sonnet-4", 
    "claude-3-5-haiku-20241022": "Claude-3.5-Haiku-20241022",
    "claude-3-5-sonnet-20241022": "Claude-3.5-Sonnet-20241022",
    "deepseek-ai_DeepSeek-R1": "DeepSeek-R1",
    "deepseek_deepseek-r1-0528": "DeepSeek-R1 (0528)",
    "deepseek-chat": "DeepSeek-V3",
    "DeepSeek-V3 (0324)": "DeepSeek-V3",
    "grok-3-mini-beta": "Grok-3-Mini (High)",
    "google_gemini-2.5-pro-preview-05-06": "Gemini-2.5-Pro-Preview-05-06",
    "google_gemma-3-27b-it": "Gemma-3-27B-Instruct",
    "Gemini-2.5-Pro-05-06": "Gemini-2.5-Pro-Preview-05-06",
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


def collect_symprompt_metrics(results_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Collect CSR, line coverage, and branch coverage metrics from symprompt unit test generation evaluations.
    
    For entries where branch_coverage is -1, exclude from branch coverage calculation but include CSR and line coverage.
    """
    
    symprompt_dir = results_dir / "unit_test_generation" / "symprompt" / "evaluations"
    if not symprompt_dir.exists():
        return {}
    
    model_metrics = defaultdict(lambda: {"csr": [], "line_coverage": [], "branch_coverage": []})
    
    for jsonl_file in symprompt_dir.glob("*.jsonl"):
        model_name = normalize_model_name(jsonl_file.stem)
        records = read_jsonl(jsonl_file)
        
        for record in records:
            # Extract metrics from the first element in metrics array
            metrics = record.get("metrics", [])
            if metrics and isinstance(metrics, list) and len(metrics) > 0:
                first_metric = metrics[0]
                if isinstance(first_metric, dict):
                    # Extract all three metrics
                    csr = first_metric.get("csr")
                    line_coverage = first_metric.get("line_coverage")
                    branch_coverage = first_metric.get("branch_coverage")
                    
                    # Always include CSR and line coverage if valid
                    if isinstance(csr, (int, float)) and isinstance(line_coverage, (int, float)):
                        model_metrics[model_name]["csr"].append(float(csr))
                        model_metrics[model_name]["line_coverage"].append(float(line_coverage))
                    
                    # Only include branch coverage if it's not -1 and is valid
                    if (isinstance(branch_coverage, (int, float)) and branch_coverage != -1):
                        model_metrics[model_name]["branch_coverage"].append(float(branch_coverage))
    
    # Calculate averages for each model
    model_averages = {}
    for model, metrics_lists in model_metrics.items():
        if metrics_lists["csr"]:  # Check if we have any valid entries
            avg_metrics = {
                "csr": sum(metrics_lists["csr"]) / len(metrics_lists["csr"]),
                "line_coverage": sum(metrics_lists["line_coverage"]) / len(metrics_lists["line_coverage"])
            }
            
            # Only calculate branch coverage average if we have valid branch coverage entries
            if metrics_lists["branch_coverage"]:
                avg_metrics["branch_coverage"] = sum(metrics_lists["branch_coverage"]) / len(metrics_lists["branch_coverage"])
            else:
                avg_metrics["branch_coverage"] = None  # No valid branch coverage data
            
            model_averages[model] = avg_metrics
    
    return model_averages


def generate_latex_table(model_scores: Dict[str, Dict[str, Optional[float]]], precision: int = 1) -> str:
    """Generate LaTeX table content with CSR, Line Coverage, and Branch Coverage columns."""
    
    # Sort models by CSR score (descending) but preserve display order for consistency
    scored_models = [(model, model_scores.get(model)) for model in MODEL_DISPLAY_ORDER 
                     if model in model_scores]
    
    # Sort by CSR score descending, with None values at the end
    scored_models.sort(key=lambda x: (-x[1]["csr"] if x[1] is not None else -999, x[0]))
    
    lines = []
    lines.append("\\subsection{Test Generation}")
    lines.append("\\begin{table*}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{0.4em}")
    lines.append("\\caption{Performance (\\%) on SymPrompt-Python unit test generation. \\textbf{Bold} indicates the best performance in each column.}")
    lines.append("\\label{tab:unit_test_generation}")
    lines.append("\\begin{tabular}{@{}c|ccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\multicolumn{3}{c}{\\textbf{SymPrompt}} \\\\")
    lines.append("\\cmidrule(l){2-4}")
    lines.append(" & \\textbf{CSR} & \\textbf{Cov\\textsubscript{L}} & \\textbf{Cov\\textsubscript{Br}} \\\\")
    lines.append("\\midrule")
    
    # Create ranking lists for each metric
    def get_ranking(metric_name):
        values = []
        for model, scores in model_scores.items():
            if metric_name == "branch_coverage" and scores[metric_name] is None:
                continue
            values.append((scores[metric_name], model))
        values.sort(reverse=True, key=lambda x: x[0])
        return values
    
    csr_ranking = get_ranking("csr")
    line_cov_ranking = get_ranking("line_coverage") 
    branch_cov_ranking = get_ranking("branch_coverage")
    
    def get_rank_style(value, ranking, metric_name):
        if metric_name == "branch_coverage" and value is None:
            return None
        
        # Find position in ranking
        for i, (score, _) in enumerate(ranking):
            if abs(score - value) < 1e-9:
                if i == 0:
                    return "first"
                elif i == 1:
                    return "second"
                elif i == 2:
                    return "third"
                break
        return None
    
    for i, (model, scores) in enumerate(scored_models):
        if scores is None:
            continue
            
        # Format each score
        csr_str = f"{scores['csr'] * 100:.{precision}f}"
        line_cov_str = f"{scores['line_coverage']:.{precision}f}"
        
        # Handle branch coverage (might be None)
        if scores['branch_coverage'] is not None:
            branch_cov_str = f"{scores['branch_coverage']:.{precision}f}"
        else:
            branch_cov_str = "N/A"
        
        # Apply ranking styles
        csr_rank = get_rank_style(scores["csr"], csr_ranking, "csr")
        line_cov_rank = get_rank_style(scores["line_coverage"], line_cov_ranking, "line_coverage")
        branch_cov_rank = get_rank_style(scores["branch_coverage"], branch_cov_ranking, "branch_coverage")
        
        # Format with appropriate styling
        if csr_rank:
            formatted_csr = f"\\cellcolor{{{csr_rank}bg}}\\textcolor{{{csr_rank}text}}{{\\textbf{{{csr_str}}}}}"
        else:
            formatted_csr = csr_str
            
        if line_cov_rank:
            formatted_line_cov = f"\\cellcolor{{{line_cov_rank}bg}}\\textcolor{{{line_cov_rank}text}}{{\\textbf{{{line_cov_str}}}}}"
        else:
            formatted_line_cov = line_cov_str
            
        if branch_cov_rank:
            formatted_branch_cov = f"\\cellcolor{{{branch_cov_rank}bg}}\\textcolor{{{branch_cov_rank}text}}{{\\textbf{{{branch_cov_str}}}}}"
        else:
            formatted_branch_cov = branch_cov_str
            
        lines.append(f"{model} & {formatted_csr} & {formatted_line_cov} & {formatted_branch_cov} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate unit test generation LaTeX table")
    parser.add_argument("--results-dir", type=Path, 
                       default=Path(__file__).parent.parent / "results",
                       help="Root directory containing evaluation results")
    parser.add_argument("--output", type=Path,
                       default=Path(__file__).parent.parent / "latex_table" / "unit_test_generation.tex",
                       help="Output LaTeX file path")
    parser.add_argument("--precision", type=int, default=1,
                       help="Decimal places for percentages")
    
    args = parser.parse_args()
    
    # Collect metrics
    model_scores = collect_symprompt_metrics(args.results_dir)
    
    if not model_scores:
        print("No unit test generation metrics found!")
        return
    
    # Generate LaTeX table
    latex_content = generate_latex_table(model_scores, args.precision)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_content + "\n", encoding="utf-8")
    
    print(f"Unit test generation table written to {args.output}")
    print(f"Found metrics for {len(model_scores)} models")


if __name__ == "__main__":
    main()