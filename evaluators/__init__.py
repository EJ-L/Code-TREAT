"""
TREAT Evaluators Package

Standalone evaluators that process raw prediction files and add metrics.
Follows the pattern: read from save/ → process → write to results/
"""

from .code_review_evaluator import evaluate_code_review_file
# from .code_generation_evaluator import evaluate_code_generation_file

# Import other evaluators when they exist  
# from .code_translation_evaluator import CodeTranslationEvaluator (already exists)

__all__ = [
    "CodeReviewEvaluator",
    "evaluate_code_review_file",
    "evaluate_code_generation_file",
    # Add other evaluators here as they're enhanced
]