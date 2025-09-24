#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results_data(results_dir):
    """Load all code_translation results and calculate metrics by model and modality"""
    model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Load HackerRank results
    hackerrank_dir = os.path.join(results_dir, 'code_translation', 'hackerrank', 'evaluations')
    if os.path.exists(hackerrank_dir):
        for filename in os.listdir(hackerrank_dir):
            if filename.endswith('.jsonl'):
                model_name = filename.replace('.jsonl', '')
                file_path = os.path.join(hackerrank_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                modality = data.get('modality', '')
                                
                                # Handle different evaluation result formats
                                if 'metrics' in data and 'pass@1' in data['metrics']:
                                    # HackerRank format with metrics
                                    pass_at_1 = data['metrics']['pass@1']
                                # elif 'evaluation_results' in data:
                                #     # PolyHumanEval format with evaluation_results
                                #     results = data['evaluation_results']
                                #     if isinstance(results, list) and len(results) > 0:
                                #         # Convert success/failure to 1.0/0.0
                                #         pass_at_1 = 1.0 if results[0].get('success', False) else 0.0
                                #     else:
                                #         pass_at_1 = 0.0
                                else:
                                    # pass_at_1 = 0.0
                                    raise ValueError(f"NO metrics or evaluation_results in {data}")
                                model_results[model_name]['hackerrank'][modality].append(pass_at_1)
                            except json.JSONDecodeError:
                                continue
    
    # Load PolyHumanEval results
    polyhumaneval_dir = os.path.join(results_dir, 'code_translation', 'polyhumaneval', 'evaluations')
    if os.path.exists(polyhumaneval_dir):
        for filename in os.listdir(polyhumaneval_dir):
            if filename.endswith('.jsonl'):
                model_name = filename.replace('.jsonl', '')
                file_path = os.path.join(polyhumaneval_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                modality = data.get('modality', '')
                                pass_at_1 = data.get('metrics', {}).get('pass@1', 0.0)
                                model_results[model_name]['polyhumaneval'][modality].append(pass_at_1)
                            except json.JSONDecodeError:
                                continue
    
    return model_results


def calculate_averages(model_results):
    """Calculate average pass@1 for each model, dataset, and modality"""
    averages = {}
    
    for model_name, datasets in model_results.items():
        averages[model_name] = {}
        
        # Calculate overall averages across all datasets
        all_python_to_java = []
        all_java_to_python = []
        all_scores = []
        
        for dataset_name, modalities in datasets.items():
            averages[model_name][dataset_name] = {}
            dataset_scores = []
            
            for modality, scores in modalities.items():
                if scores:
                    avg_score = sum(scores) / len(scores) * 100  # Convert to percentage
                    averages[model_name][dataset_name][modality] = avg_score
                    dataset_scores.extend(scores)
                    
                    # Collect for overall calculation
                    if modality == 'python->java':
                        all_python_to_java.extend(scores)
                    elif modality == 'java->python':
                        all_java_to_python.extend(scores)
                else:
                    averages[model_name][dataset_name][modality] = 0.0
            
            # Calculate dataset overall
            if dataset_scores:
                averages[model_name][dataset_name]['overall'] = sum(dataset_scores) / len(dataset_scores) * 100
            else:
                averages[model_name][dataset_name]['overall'] = 0.0
            
            all_scores.extend(dataset_scores)
        
        # Calculate overall averages across all datasets
        averages[model_name]['overall'] = {}
        if all_python_to_java:
            averages[model_name]['overall']['python->java'] = sum(all_python_to_java) / len(all_python_to_java) * 100
        else:
            averages[model_name]['overall']['python->java'] = 0.0
            
        if all_java_to_python:
            averages[model_name]['overall']['java->python'] = sum(all_java_to_python) / len(all_java_to_python) * 100
        else:
            averages[model_name]['overall']['java->python'] = 0.0
            
        # Overall overall average
        if all_scores:
            averages[model_name]['overall']['overall'] = sum(all_scores) / len(all_scores) * 100
            # print(model_name)
            # print(sum(all_scores))
            # print(len(all_scores))
        else:
            averages[model_name]['overall']['overall'] = 0.0
    
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
    
    # Column keys for the table
    columns = [
        'overall_overall',
        'overall_python->java', 
        'overall_java->python',
        'hackerrank_overall',
        'hackerrank_python->java',
        'hackerrank_java->python',
        'polyhumaneval_overall',
        'polyhumaneval_python->java',
        'polyhumaneval_java->python'
    ]
    
    for column in columns:
        column_scores = []
        for model_name, datasets in averages.items():
            if column.startswith('overall_'):
                _, modality = column.split('_', 1)
                score = datasets.get('overall', {}).get(modality, 0)
            else:
                dataset, modality = column.split('_', 1)
                score = datasets.get(dataset, {}).get(modality, 0)
            
            column_scores.append(score)
        
        # Sort scores in descending order and take top 3
        column_scores.sort(reverse=True)
        top_scores[column] = column_scores[:3]
    
    return top_scores


def update_latex_table(template_path, output_path, averages):
    """Update the LaTeX table with new results"""
    
    # Read the template
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the table data section (between \midrule and \bottomrule)
    start_marker = "\\midrule"
    end_marker = "\\bottomrule"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find table markers in the LaTeX file")
    
    # Find top 3 scores for formatting
    top_scores = find_top_scores(averages)
    
    # Sort models by overall performance (descending)
    sorted_models = sorted(averages.items(), 
                          key=lambda x: x[1].get('overall', {}).get('overall', 0), 
                          reverse=True)
    
    # Generate new table rows
    new_rows = []
    for model_name, datasets in sorted_models:
        # Clean up model name for display
        display_name = model_name.replace('_', ' ').replace('-', '-')
        
        # Get scores for each column in the correct order
        overall_overall = datasets.get('overall', {}).get('overall', 0)
        overall_py_java = datasets.get('overall', {}).get('python->java', 0)
        overall_java_py = datasets.get('overall', {}).get('java->python', 0)
        
        hr_overall = datasets.get('hackerrank', {}).get('overall', 0)
        hr_py_java = datasets.get('hackerrank', {}).get('python->java', 0)
        hr_java_py = datasets.get('hackerrank', {}).get('java->python', 0)
        
        poly_overall = datasets.get('polyhumaneval', {}).get('overall', 0)
        poly_py_java = datasets.get('polyhumaneval', {}).get('python->java', 0)
        poly_java_py = datasets.get('polyhumaneval', {}).get('java->python', 0)
        
        # Format with top 3 highlighting
        row = f"    {display_name} & " + \
              f"{format_score(overall_overall, top_scores, 'overall_overall')} & " + \
              f"{format_score(overall_py_java, top_scores, 'overall_python->java')} & " + \
              f"{format_score(overall_java_py, top_scores, 'overall_java->python')} & " + \
              f"{format_score(hr_overall, top_scores, 'hackerrank_overall')} & " + \
              f"{format_score(hr_py_java, top_scores, 'hackerrank_python->java')} & " + \
              f"{format_score(hr_java_py, top_scores, 'hackerrank_java->python')} & " + \
              f"{format_score(poly_overall, top_scores, 'polyhumaneval_overall')} & " + \
              f"{format_score(poly_py_java, top_scores, 'polyhumaneval_python->java')} & " + \
              f"{format_score(poly_java_py, top_scores, 'polyhumaneval_java->python')} \\\\"
        
        new_rows.append(row)
    
    # Replace the table content
    new_content = content[:start_idx + len(start_marker)] + "\n" + \
                  "\n".join(new_rows) + "\n    " + \
                  content[end_idx:]
    
    # Write the updated table
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated LaTeX table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Update code_translation LaTeX table with latest results')
    parser.add_argument('--results-dir', default='results', help='Results directory (default: results)')
    parser.add_argument('--template', default='latex_table/code_translation.tex', help='Template LaTeX file')
    parser.add_argument('--output', help='Output LaTeX file (default: same as template)')
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.template
    
    # Load and process results
    print("Loading results data...")
    model_results = load_results_data(args.results_dir)
    
    print("Calculating averages...")
    averages = calculate_averages(model_results)
    
    print(f"Found results for {len(averages)} models")
    
    # Update the table
    print("Updating LaTeX table...")
    update_latex_table(args.template, args.output, averages)
    
    print("Done!")


if __name__ == '__main__':
    main()