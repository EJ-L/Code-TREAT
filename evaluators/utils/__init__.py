"""
TREAT Evaluation Utilities

Shared utilities for evaluation tasks across different benchmarks.
"""

# Import from utils module
from .utils import (
    normalize_unicode_characters,
    get_optimal_worker_count,
    split_prediction_file_by_dataset,
    save_evaluation_results,
    load_prediction_file,
    create_results_directory,
    format_evaluation_summary
)

# Import from compression_utils module
from .compression_utils import (
    compress_test_cases,
    decompress_test_cases
)

# Import from pass_at_k module
from .pass_at_k import (
    compute_pass_at_k_from_results
)

# Import from executor module
from .executor import (
    CodeExecutor
)

__all__ = [
    # Text processing utilities
    "normalize_unicode_characters",
    
    # Performance and system utilities
    "get_optimal_worker_count", 
    
    # File processing utilities
    "split_prediction_file_by_dataset",
    "save_evaluation_results",
    "load_prediction_file",
    "create_results_directory",
    "format_evaluation_summary",
    
    # Compression utilities
    "compress_test_cases",
    "decompress_test_cases",
    
    # Evaluation metrics
    "compute_pass_at_k_from_results",
    
    # Code execution
    "CodeExecutor"
]