#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_code_reasoning_results(results_dir):
    """Load input_prediction and output_prediction results and calculate averages by model and language"""
    model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Tasks to process
    tasks = ['input_prediction', 'output_prediction']
    datasets = ['geeksforgeeks', 'hackerrank']
    
    for task in tasks:
        for dataset in datasets:
            eval_dir = os.path.join(results_dir, task, dataset, 'evaluations')
            if os.path.exists(eval_dir):
                for filename in os.listdir(eval_dir):
                    if filename.endswith('.jsonl'):
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
                                        
                                        model_results[model_name][task][lang].append(score)
                                    except json.JSONDecodeError:
                                        continue
    
    return model_results


def calculate_averages(model_results):
    """Calculate average pass@1 scores for each model, task, and language"""
    averages = {}
    
    for model_name, tasks in model_results.items():
        averages[model_name] = {}
        
        for task, languages in tasks.items():
            for lang, scores in languages.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    task_lang_key = f"{task}_{lang}"
                    averages[model_name][task_lang_key] = avg_score
                else:
                    task_lang_key = f"{task}_{lang}"
                    averages[model_name][task_lang_key] = 0.0
    
    # Calculate overall averages
    for model_name, model_data in averages.items():
        # Input overall (average of input_prediction_python and input_prediction_java)
        input_python = model_data.get('input_prediction_python', 0)
        input_java = model_data.get('input_prediction_java', 0)
        input_scores = [s for s in [input_python, input_java] if s > 0]
        averages[model_name]['input_overall'] = sum(input_scores) / len(input_scores) if input_scores else 0
        
        # Output overall (average of output_prediction_python and output_prediction_java)
        output_python = model_data.get('output_prediction_python', 0)
        output_java = model_data.get('output_prediction_java', 0)
        output_scores = [s for s in [output_python, output_java] if s > 0]
        averages[model_name]['output_overall'] = sum(output_scores) / len(output_scores) if output_scores else 0
        
        # Overall Python (average of input_prediction_python and output_prediction_python)
        python_scores = [s for s in [input_python, output_python] if s > 0]
        averages[model_name]['overall_python'] = sum(python_scores) / len(python_scores) if python_scores else 0
        
        # Overall Java (average of input_prediction_java and output_prediction_java)
        java_scores = [s for s in [input_java, output_java] if s > 0]
        averages[model_name]['overall_java'] = sum(java_scores) / len(java_scores) if java_scores else 0
        
        # Overall Overall (average across all 4 columns)
        all_scores = [s for s in [input_python, input_java, output_python, output_java] if s > 0]
        averages[model_name]['overall_overall'] = sum(all_scores) / len(all_scores) if all_scores else 0
    
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
    
    # Column keys match the table structure: Overall Overall, Overall Python, Overall Java, Input Overall, Input Python, Input Java, Output Overall, Output Python, Output Java
    columns = ['overall_overall', 'overall_python', 'overall_java', 'input_overall', 'input_prediction_python', 'input_prediction_java', 'output_overall', 'output_prediction_python', 'output_prediction_java']
    
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
        'Claude-3.5-Sonnet-20241022': 'Claude-3.5-Sonnet',
        'Claude-3.5-Haiku-20241022': 'Claude-3.5-Haiku',
        'Claude-3.7-Sonnet': 'Claude-3.7-Sonnet',
        'Claude-Sonnet-4': 'Claude-Sonnet-4',
        'Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B',
        'Llama-3.1-70B-Instruct': 'LLaMA-3.1-70B',
        'Llama-3.1-8B-Instruct': 'LLaMA-3.1-8B',
        'Llama-4-Scout-17B-16E-Instruct': 'LLaMA-4-Scout',
        'o3-mini (Med)': 'o3-mini (Med)',
        'o4-mini (Med)': 'o4-mini (Med)',
        'GPT-4o-2024-11-20': 'GPT-4o',
        'GPT-4-turbo-2024-04-09': 'GPT-4-turbo',
        'GPT-4.1-2025-04-14': 'GPT-4.1',
        'GPT-3.5-turbo-0125': 'GPT-3.5-turbo',
        'Gemini-2.5-Pro-Preview-05-06': 'Gemini-2.5-Pro-05-06',
        'Gemma-3-27B-Instruct': 'Gemma-3-27B-Instruct',
        'Grok-3-Mini (High)': 'Grok-3-Mini',
        'DeepSeek-V3': 'DeepSeek-V3',
        'DeepSeek-R1': 'DeepSeek-R1',
        'DeepSeek-R1 (0528)': 'DeepSeek-R1 (0528)',
        'Qwen2.5-72B-Instruct': 'Qwen2.5-72B',
        'Qwen2.5-Coder-32B-Instruct': 'Qwen2.5-Coder-32B',
        'Qwen3-235B-A22B': 'Qwen3-235B-A22B',
        'Qwen3-30B-A3B': 'Qwen3-30B-A3B',
        'Qwen3-32B': 'Qwen3-32B',
        'GPT-5': 'GPT-5'
    }
    
    return name_mappings.get(model_name, model_name)


def create_latex_table(output_path, averages):
    """Create a new LaTeX table for code reasoning results"""
    
    # Find top 3 scores for formatting
    top_scores = find_top_scores(averages)
    
    # Calculate overall average for sorting (average across all 4 columns)
    model_overall = {}
    for model_name, model_data in averages.items():
        scores = []
        for column in ['input_prediction_python', 'input_prediction_java', 'output_prediction_python', 'output_prediction_java']:
            score = model_data.get(column, 0)
            if score > 0:  # Only include non-zero scores in average
                scores.append(score)
        model_overall[model_name] = sum(scores) / len(scores) if scores else 0
    
    # Sort models by overall overall performance (descending)
    sorted_models = sorted(averages.items(), 
                          key=lambda x: x[1].get('overall_overall', 0), 
                          reverse=True)
    
    # Generate table content
    content = f"""\\subsection{{Code Reasoning}}

\\begin{{table*}}[ht]
\\centering
\\small
\\renewcommand{{\\arraystretch}}{{1.2}}
\\setlength{{\\tabcolsep}}{{0.5em}}
\\caption{{Input and Output Prediction Pass@1 Accuracy (\\%) by Model}}
\\label{{tab:code_reasoning}}
\\begin{{tabular}}{{@{{}}cccccccccc@{{}}}}
\\toprule
\\textbf{{Model}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Overall}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Input}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Output}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(l){{8-10}}
 & \\textbf{{Overall}} & \\textbf{{Python}} & \\textbf{{Java}} & \\textbf{{Overall}} & \\textbf{{Python}} & \\textbf{{Java}} & \\textbf{{Overall}} & \\textbf{{Python}} & \\textbf{{Java}} \\\\
\\midrule
"""
    
    # Generate table rows
    for model_name, model_data in sorted_models:
        # Clean up model name for display
        display_name = clean_model_name(model_name)
        
        # Get formatted scores for each column
        overall_overall = format_score(model_data.get('overall_overall', 0), top_scores, 'overall_overall')
        overall_python = format_score(model_data.get('overall_python', 0), top_scores, 'overall_python')
        overall_java = format_score(model_data.get('overall_java', 0), top_scores, 'overall_java')
        input_overall = format_score(model_data.get('input_overall', 0), top_scores, 'input_overall')
        input_python = format_score(model_data.get('input_prediction_python', 0), top_scores, 'input_prediction_python')
        input_java = format_score(model_data.get('input_prediction_java', 0), top_scores, 'input_prediction_java')
        output_overall = format_score(model_data.get('output_overall', 0), top_scores, 'output_overall')
        output_python = format_score(model_data.get('output_prediction_python', 0), top_scores, 'output_prediction_python')
        output_java = format_score(model_data.get('output_prediction_java', 0), top_scores, 'output_prediction_java')
        
        row = f"{display_name} & {overall_overall} & {overall_python} & {overall_java} & {input_overall} & {input_python} & {input_java} & {output_overall} & {output_python} & {output_java} \\\\"
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
    parser = argparse.ArgumentParser(description='Create code reasoning LaTeX table with latest results')
    parser.add_argument('--results-dir', default='results', help='Results directory (default: results)')
    parser.add_argument('--output', default='latex_table/code_reasoning.tex', help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Load and process results
    print("Loading code reasoning results...")
    model_results = load_code_reasoning_results(args.results_dir)
    
    print("Calculating averages...")
    averages = calculate_averages(model_results)
    
    print(f"Found results for {len(averages)} models")
    
    # Create the table
    print("Creating LaTeX table...")
    create_latex_table(args.output, averages)
    
    print("Done!")


if __name__ == '__main__':
    main()