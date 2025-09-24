import math
from typing import Dict, List

def _prediction_passed(res: dict) -> bool:
    """
    A prediction is 'passed' iff all of its test cases passed.
    Accept either 'success' or 'passed' as the truthy key per test case.
    """
    # Case 1: executor returns a list of per-testcase results
    if "results" in res and isinstance(res["results"], list):
        if not res["results"]:
            return False  # no tests => treat as failed
        return all(
            bool(case.get("success") or case.get("passed"))
            for case in res["results"]
        )
    # Case 2: flat result
    return bool(res.get("success") or res.get("passed"))

def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """
    HumanEval unbiased pass@k estimator for a single item given:
      n = number of predictions, c = number of correct predictions.
    """
    if k <= 0 or n <= 0:
        return 0.0
    if k > n:
        k = n
    # If fewer than k failures exist, at least one success appears in any k-sample.
    if (n - c) < k:
        return 1.0
    # Guard against comb domain errors (e.g., if k > n after adjustments)
    denom = math.comb(n, k) if 0 <= k <= n else 0
    if denom == 0:
        return 0.0
    return 1.0 - (math.comb(n - c, k) / denom)

def compute_pass_at_k_from_results(results: List[dict], k_list: List[int]) -> Dict[str, float]:
    """
    Compute pass@k for a *single problem* from its prediction results.

    Args:
        results: one dict per prediction attempt from EXECUTOR.run(...)
        k_list: list of k values to compute (e.g., [1, 5, 10])

    Returns:
        Dict like {"pass@1": 0.0, "pass@5": 1.0, ...}
    """
    n = len(results)
    passed_flags = [ _prediction_passed(r) for r in results ]
    c = sum(passed_flags)

    metrics = {}
    for k in k_list:
        metrics[f"pass@{k}"] = _unbiased_pass_at_k(n, c, k)
    return metrics