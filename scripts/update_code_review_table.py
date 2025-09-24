#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_code_review_results(results_dir):
    """Load code_review_generation results and calculate metrics by model and language"""
    model_results = defaultdict(lambda: defaultdict(list))
    
    # Load GitHub 2023 results
    github_dir = os.path.join(results_dir, 'code_review_generation', 'github_2023', 'evaluations')
    if os.path.exists(github_dir):
        for filename in os.listdir(github_dir):
            if filename.endswith('.jsonl'):
                model_name = filename.replace('.jsonl', '')
                file_path = os.path.join(github_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                lang = data.get('lang', '')
                                
                                # Extract GPT-4o score (handle both list and dict formats)
                                gpt4o_data = data.get('metrics', {}).get('gpt-4o', [])
                                
                                # Handle inconsistent data structure
                                if isinstance(gpt4o_data, list) and gpt4o_data:
                                    score = (gpt4o_data[0] / 5.0) * 100  # Convert to percentage (1-5 scale)
                                    model_results[model_name][lang].append(score)
                                elif isinstance(gpt4o_data, dict) and 'gpt-4o' in gpt4o_data:
                                    gpt4o_scores = gpt4o_data['gpt-4o']
                                    if gpt4o_scores and len(gpt4o_scores) > 0:
                                        score = (gpt4o_scores[0] / 5.0) * 100  # Convert to percentage (1-5 scale)
                                        model_results[model_name][lang].append(score)
                            except json.JSONDecodeError:
                                continue
    
    return model_results


def calculate_averages(model_results):
    """Calculate average scores for each model and language"""
    averages = {}
    
    for model_name, languages in model_results.items():
        averages[model_name] = {}
        
        # Calculate averages for each language
        all_scores = []
        for lang, scores in languages.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                averages[model_name][lang] = avg_score
                all_scores.extend(scores)
            else:
                averages[model_name][lang] = 0.0
        
        # Calculate overall average
        if all_scores:
            averages[model_name]['overall'] = sum(all_scores) / len(all_scores)
        else:
            averages[model_name]['overall'] = 0.0
    
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
    """Find the top 3 scores for each column (overall + each language)"""
    top_scores = {}
    
    # Languages found in the data
    all_languages = set()
    for model_name, model_langs in averages.items():
        all_languages.update(model_langs.keys())
    
    languages = ['overall'] + sorted([lang for lang in all_languages if lang != 'overall'])
    
    for lang in languages:
        lang_scores = []
        for model_name, model_langs in averages.items():
            score = model_langs.get(lang, 0)
            lang_scores.append(score)
        
        # Sort scores in descending order and take top 3
        lang_scores.sort(reverse=True)
        top_scores[lang] = lang_scores[:3]
    
    return top_scores


def clean_model_name(model_name):
    """Clean up model name for display in table"""
    # Handle common model name patterns
    name_mappings = {
        'Claude-3.5-Sonnet-20241022': 'Claude-3-5-Sonnet-20241022',
        'Claude-3.5-Haiku-20241022': 'Claude-3-5-Haiku-20241022',
        'Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
        'Llama-3.1-70B-Instruct': 'LLaMA-3.1-70B-Instruct',
        'Llama-3.1-8B-Instruct': 'LLaMA-3.1-8B-Instruct',
        'Llama-4-Scout-17B-16E-Instruct': 'LLaMA-4-Scout-17B-16E-Instruct',
        'o3-mini (Med)': 'o3-mini (Med)',
        'o4-mini (Med)': 'o4-mini (Med)',
        'GPT-4o-2024-11-20': 'GPT-4o-2024-11-20',
        'GPT-4-turbo-2024-04-09': 'GPT-4-turbo-2024-04-09',
        'GPT-4.1-2025-04-14': 'GPT-4.1-2025-04-14',
        'GPT-3.5-turbo-0125': 'GPT-3.5-turbo-0125',
        'Gemini-2.5-Pro-05-06': 'Gemini-2.5-Pro-05-06'
    }
    
    return name_mappings.get(model_name, model_name)


def create_latex_table(output_path, averages):
    """Create a new LaTeX table for code review results"""
    
    # Find top 3 scores for formatting
    top_scores = find_top_scores(averages)
    
    # Sort models by overall performance (descending)
    sorted_models = sorted(averages.items(), 
                          key=lambda x: x[1].get('overall', 0), 
                          reverse=True)
    
    # Get all languages for column headers
    all_languages = set()
    for model_name, model_langs in averages.items():
        all_languages.update(model_langs.keys())
    
    languages = sorted([lang for lang in all_languages if lang != 'overall'])
    
    # Generate column headers
    lang_columns = ' & '.join([f'\\textbf{{{lang.title()}}}' for lang in languages])
    
    # Generate table content
    content = f"""\\subsection{{Code Review}}
\\begin{{table}}[ht]
  \\centering
  \\small
  \\setlength{{\\tabcolsep}}{{0.3em}}
  \\caption{{Code review generation accuracy (\\%) by model and programming language. \\textbf{{Bold}} indicates the best performance in each column.}}
  \\label{{tab:code_review}}
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{@{{}}*{{{len(languages) + 2}}}{{c}}@{{}}}}
    \\toprule
    \\textbf{{Model}} & \\textbf{{Overall}} & {lang_columns} \\\\
    \\midrule
"""
    
    # Generate table rows
    for model_name, languages_data in sorted_models:
        # Skip human baseline - we'll handle it separately if needed
        if 'human' in model_name.lower() or 'baseline' in model_name.lower():
            continue
            
        # Clean up model name for display
        display_name = clean_model_name(model_name)
        
        # Get scores for each language column
        overall = languages_data.get('overall', 0)
        row_scores = [format_score(overall, top_scores, 'overall')]
        
        for lang in languages:
            score = languages_data.get(lang, 0)
            row_scores.append(format_score(score, top_scores, lang))
        
        row = f"    {display_name} & " + " & ".join(row_scores) + "\\\\"
        content += row + '\n\n'
        # content += "\n"  # Add blank line between models for readability
    
    content += """    \\bottomrule
  \\end{tabular}
}
\\end{table}
"""
    
    # Write the table
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"LaTeX table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create code review LaTeX table with latest results')
    parser.add_argument('--results-dir', default='results', help='Results directory (default: results)')
    parser.add_argument('--output', default='latex_table/code_review.tex', help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Load and process results
    print("Loading code review results...")
    model_results = load_code_review_results(args.results_dir)
    
    print("Calculating averages...")
    averages = calculate_averages(model_results)
    
    print(f"Found results for {len(averages)} models")
    
    # Create the table
    print("Creating LaTeX table...")
    create_latex_table(args.output, averages)
    
    print("Done!")


if __name__ == '__main__':
    main()