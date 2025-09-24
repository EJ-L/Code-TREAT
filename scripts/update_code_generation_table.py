#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_code_generation_results(results_dir):
    """Load code generation results and calculate averages by model and language"""
    model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Datasets to process
    datasets = ['geeksforgeeks', 'hackerrank']
    
    for dataset in datasets:
        eval_dir = os.path.join(results_dir, 'code_generation', dataset, 'evaluations')
        if os.path.exists(eval_dir):
            for filename in os.listdir(eval_dir):
                if filename.endswith('.jsonl') and 'backup' not in filename.lower():
                    model_name = filename.replace('.jsonl', '')
                    file_path = os.path.join(eval_dir, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    lang = data.get('lang', '')
                                    
                                    # Extract pass@1 score
                                    pass_at_1 = data.get('metrics', {}).get('pass@1', 0)
                                    score = pass_at_1 * 100  # Convert to percentage
                                    
                                    model_results[model_name][lang][dataset].append(score)
                                except json.JSONDecodeError:
                                    continue
    
    return model_results


# def calculate_averages(model_results):
#     """Calculate average pass@1 scores for each model, language, and dataset"""
#     averages = {}
    
#     for model_name, languages in model_results.items():
#         averages[model_name] = {}
        
#         for lang, datasets in languages.items():
#             for dataset, scores in datasets.items():
#                 if scores:
#                     avg_score = sum(scores) / len(scores)
#                     dataset_key = f"{lang}_{dataset}"
#                     averages[model_name][dataset_key] = avg_score
#                 else:
#                     dataset_key = f"{lang}_{dataset}"
#                     averages[model_name][dataset_key] = 0.0
    
#     # Calculate overall averages for each language
#     for model_name, model_data in averages.items():
#         # Python overall (average of python_geeksforgeeks and python_hackerrank)
#         python_geeks = model_data.get('python_geeksforgeeks', 0)
#         python_hr = model_data.get('python_hackerrank', 0)
#         python_scores = [s for s in [python_geeks, python_hr] if s > 0]
#         averages[model_name]['python_overall'] = sum(python_scores) / len(python_scores) if python_scores else 0
        
#         # Java overall (average of java_geeksforgeeks and java_hackerrank)
#         java_geeks = model_data.get('java_geeksforgeeks', 0)
#         java_hr = model_data.get('java_hackerrank', 0)
#         java_scores = [s for s in [java_geeks, java_hr] if s > 0]
#         averages[model_name]['java_overall'] = sum(java_scores) / len(java_scores) if java_scores else 0
        
#         # Combined overall (average of python_overall and java_overall)
#         python_overall = averages[model_name]['python_overall']
#         java_overall = averages[model_name]['java_overall']
#         combined_scores = [s for s in [python_overall, java_overall] if s > 0]
#         averages[model_name]['combined_overall'] = sum(combined_scores) / len(combined_scores) if combined_scores else 0
        
#         # Overall GeeksforGeeks (average of python_geeksforgeeks and java_geeksforgeeks)
#         python_geeks = averages[model_name]['python_geeksforgeeks']
#         java_geeks = averages[model_name]['java_geeksforgeeks']
#         geeks_scores = [s for s in [python_geeks, java_geeks] if s > 0]
#         averages[model_name]['overall_geeksforgeeks'] = sum(geeks_scores) / len(geeks_scores) if geeks_scores else 0
        
#         # Overall HackerRank (average of python_hackerrank and java_hackerrank)
#         python_hr = averages[model_name]['python_hackerrank']
#         java_hr = averages[model_name]['java_hackerrank']
#         hr_scores = [s for s in [python_hr, java_hr] if s > 0]
#         averages[model_name]['overall_hackerrank'] = sum(hr_scores) / len(hr_scores) if hr_scores else 0
    
#     return averages

def calculate_averages(model_results):
    averages = {}

    for model_name, langs in model_results.items():
        averages[model_name] = {}
        total_sum, total_count = 0, 0
        lang_sum, lang_count = defaultdict(int), defaultdict(int)
        ds_sum, ds_count = defaultdict(int), defaultdict(int)

        # record per (lang,dataset)
        for lang, datasets in langs.items():
            for dataset, scores in datasets.items():
                s = sum(scores)
                n = len(scores)
                if n == 0:
                    avg = 0.0
                else:
                    avg = s / n
                averages[model_name][f"{lang}_{dataset}"] = avg
                # accumulate
                total_sum += s; total_count += n
                lang_sum[lang] += s; lang_count[lang] += n
                ds_sum[dataset] += s; ds_count[dataset] += n

        # language overall
        for lang in lang_sum:
            averages[model_name][f"{lang}_overall"] = (
                lang_sum[lang] / lang_count[lang] if lang_count[lang] else 0.0
            )
        # dataset overall
        for ds in ds_sum:
            averages[model_name][f"overall_{ds}"] = (
                ds_sum[ds] / ds_count[ds] if ds_count[ds] else 0.0
            )
        # combined overall
        averages[model_name]["combined_overall"] = (
            total_sum / total_count if total_count else 0.0
        )
    return averages

def format_score(score, top_scores, column_key):
    """Format score with appropriate markup for top 3 scores in the column"""
    column_scores = top_scores.get(column_key, [])
    
    if len(column_scores) >= 1 and abs(score - column_scores[0]) < 0.01:
        return f"\\cellcolor{{firstbg}}\\textcolor{{firsttext}}{{\\textbf{{{score:.1f}}}}}"
    elif len(column_scores) >= 2 and abs(score - column_scores[1]) < 0.01:
        return f"\\cellcolor{{secondbg}}\\textcolor{{secondtext}}{{\\textbf{{{score:.1f}}}}}"
    elif len(column_scores) >= 3 and abs(score - column_scores[2]) < 0.01:
        return f"\\cellcolor{{thirdbg}}\\textcolor{{thirdtext}}{{\\textbf{{{score:.1f}}}}}"
    else:
        return f"{score:.1f}"


def find_top_scores(averages):
    """Find the top 3 scores for each column"""
    top_scores = {}
    
    # Column keys match the table structure
    columns = ['combined_overall', 'overall_geeksforgeeks', 'overall_hackerrank', 'python_overall', 'python_geeksforgeeks', 'python_hackerrank', 'java_overall', 'java_geeksforgeeks', 'java_hackerrank']
    
    for column in columns:
        column_scores = []
        for model_name, model_data in averages.items():
            score = model_data.get(column, 0)
            column_scores.append(score)
        
        # Sort scores in descending order and take top 3
        column_scores.sort(reverse=True)
        top_scores[column] = column_scores[:3]
    
    return top_scores


def clean_model_name(model_name):
    """Clean up model name for display in table"""
    # Handle common model name patterns
    name_mappings = {
        'Claude-3.5-Sonnet-20241022': 'Claude-3.5-Sonnet-20241022',
        'Claude-3.5-Haiku-20241022': 'Claude-3.5-Haiku-20241022',
        'Claude-3.7-Sonnet': 'Claude-3.7-Sonnet',
        'Claude-Sonnet-4': 'Claude-Sonnet-4',
        'LLaMA-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
        'LLaMA-3.1-70B-Instruct': 'LLaMA-3.1-70B-Instruct',
        'LLaMA-3.1-8B-Instruct': 'LLaMA-3.1-8B-Instruct',
        'LLaMA-4-Scout-17B-16E-Instruct': 'LLaMA-4-Scout-17B-16E-Instruct',
        'o3-mini (Med)': 'o3-mini (Med)',
        'o4-mini (Med)': 'o4-mini (Med)',
        'GPT-4o-2024-11-20': 'GPT-4o-2024-11-20',
        'GPT-4-turbo-2024-04-09': 'GPT-4-turbo-2024-04-09',
        'GPT-4.1-2025-04-14': 'GPT-4.1-2025-04-14',
        'GPT-3.5-turbo-0125': 'GPT-3.5-turbo-0125',
        'Gemini-2.5-Pro-Preview-05-06': 'Gemini-2.5-Pro-Preview-05-06',
        'Gemma-3-27B-Instruct': 'Gemma-3-27B-Instruct',
        'Grok-3-Mini (High)': 'Grok-3-Mini (High)',
        'DeepSeek-V3': 'DeepSeek-V3',
        'DeepSeek-R1': 'DeepSeek-R1',
        'DeepSeek-R1 (0528)': 'DeepSeek-R1 (0528)',
        'Qwen2.5-72B-Instruct': 'Qwen2.5-72B-Instruct',
        'Qwen2.5-Coder-32B-Instruct': 'Qwen2.5-Coder-32B-Instruct',
        'Qwen3-235B-A22B': 'Qwen3-235B-A22B',
        'Qwen3-30B-A3B': 'Qwen3-30B-A3B',
        'Qwen3-32B': 'Qwen3-32B',
        'GPT-5': 'GPT-5'
    }
    
    return name_mappings.get(model_name, model_name)


def create_latex_table(output_path, averages):
    """Create a new LaTeX table for code generation results"""
    
    # Find top scores for formatting
    top_scores = find_top_scores(averages)
    
    # Sort models by combined overall performance (descending)
    sorted_models = sorted(averages.items(), 
                          key=lambda x: x[1].get('combined_overall', 0), 
                          reverse=True)
    
    # Generate table content
    content = f"""\\subsection{{Code Generation}}
\\begin{{table*}}[h]
\\centering
\\tiny
\\setlength{{\\tabcolsep}}{{0.2em}}
\\caption{{Pass@1 accuracy (\\%) by model, split into Python and Java subsets. \\textbf{{Bold}} indicates the best performance in each column.}}
\\label{{tab:pass1_split}}
\\begin{{tabular}}{{@{{}}cccc|ccc|ccc@{{}}}}
\\toprule
\\textbf{{Model}} 
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Overall}}}}
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Python}}}} 
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Java}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(l){{8-10}}
 & \\textbf{{Overall}} & \\textbf{{GeeksforGeeks}} & \\textbf{{HackerRank}}
 & \\textbf{{Overall}} & \\textbf{{GeeksforGeeks}} & \\textbf{{HackerRank}} 
 & \\textbf{{Overall}} & \\textbf{{GeeksforGeeks}} & \\textbf{{HackerRank}} \\\\
\\midrule
"""
    
    # Generate table rows
    for model_name, model_data in sorted_models:
        # Clean up model name for display
        display_name = clean_model_name(model_name)
        
        # Get formatted scores for each column
        combined_overall = format_score(model_data.get('combined_overall', 0), top_scores, 'combined_overall')
        overall_geeks = format_score(model_data.get('overall_geeksforgeeks', 0), top_scores, 'overall_geeksforgeeks')
        overall_hr = format_score(model_data.get('overall_hackerrank', 0), top_scores, 'overall_hackerrank')
        python_overall = format_score(model_data.get('python_overall', 0), top_scores, 'python_overall')
        python_geeks = format_score(model_data.get('python_geeksforgeeks', 0), top_scores, 'python_geeksforgeeks')
        python_hr = format_score(model_data.get('python_hackerrank', 0), top_scores, 'python_hackerrank')
        java_overall = format_score(model_data.get('java_overall', 0), top_scores, 'java_overall')
        java_geeks = format_score(model_data.get('java_geeksforgeeks', 0), top_scores, 'java_geeksforgeeks')
        java_hr = format_score(model_data.get('java_hackerrank', 0), top_scores, 'java_hackerrank')
        
        row = f"{display_name} & {combined_overall} & {overall_geeks} & {overall_hr} & {python_overall} & {python_geeks} & {python_hr} & {java_overall} & {java_geeks} & {java_hr} \\\\"
        content += row + "\n"
    
    content += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Write the table
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"LaTeX table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create code generation LaTeX table with latest results')
    parser.add_argument('--results-dir', default='results', help='Results directory (default: results)')
    parser.add_argument('--output', default='latex_table/code_generation.tex', help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Load and process results
    print("Loading code generation results...")
    model_results = load_code_generation_results(args.results_dir)
    
    print("Calculating averages...")
    averages = calculate_averages(model_results)
    
    print(f"Found results for {len(averages)} models")
    
    # Create the table
    print("Creating LaTeX table...")
    create_latex_table(args.output, averages)
    
    print("Done!")


if __name__ == '__main__':
    main()