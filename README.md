# TREAT: Code LLMs Trustworthiness / Reliability Evaluation And Testing

## Overview

Large foundation models are fundamentally transforming the software engineering landscape, demonstrating exceptional potential across diverse tasks including code generation, debugging, and testing.  
However, despite this rapid progress, a significant gap remains in comprehensive and rigorous evaluation methodologies for assessing the **trustworthiness** and **reliability** of these models across real-world software engineering scenarios.

Existing benchmarks suffer from limited task scope and fail to incorporate critical evaluation aspects such as multi-modality coding abilities and robustness of models.  

To bridge this gap, we present an evaluation framework called **TREAT** (*Code LLMs **T**rustworthiness / **R**eliability **E**valuation **A**nd **T**esting*) that provides a **holistic assessment** of model performance in code intelligence tasks.

---

## Key Contributions

TREAT addresses key limitations in existing approaches with four main improvements:

1. **Multi-Task Holistic Evaluation**  
   Covers diverse software engineering activities beyond narrow coding challenges.  

2. **Multi-Language and Multi-Modality Assessment**  
   Extends beyond single-language, text-only benchmarks to include multi-modality coding tasks.  

3. **Robustness Assessment**  
   Evaluates model reliability under semantically-preserving code transformations.  

4. **Rigorous Evaluation Methodology**  
   Enhances trustworthiness of evaluation results through diverse prompts and adaptive solution extraction.  

---

## Insights from Evaluations

Based on this framework, we evaluated over **25 state-of-the-art models** and uncovered key insights:

- 📌 Current models show substantial **performance variation across programming tasks**.  
- 📌 **Multi-modal language models** demonstrate limitations in UI code generation and modification.  
- 📌 Existing models exhibit **severe robustness issues** on coding tasks.  
- 📌 Our **multi-prompt evaluation method** mitigates prompt bias and yields more reliable results.  

---

## Leaderboard Results
We cover the overall model performance across Code Generation (CG), Code Summarization (CS), Code Translation (CT), Code Reasoning (CR), Code Review Generation (CRv), Test Generation (TG), and Vulnerability Detection (VD).

# Leaderboard: Overall model performance (%) on general coding tasks

| Rank | Model Name          | CG     | CS     | CT     | CR     | CRv   | TG     | VD     |
|------|---------------------|--------|--------|--------|--------|-------|--------|--------|
| 1    | GPT-5               | 🥇89.9 | 🥇98.4 | 🥇97.9 | 🥈97.8 | 26.9  | 🥇82.6 | 🥈67.3 |
| 2    | Claude-Sonnet-4     | 74.0   | 93.8   | 86.0   | 87.9   | 30.9  | 🥉77.0 | 🥇69.5 |
| 3    | DeepSeek-R1 (0528)  | 68.8   | 90.6   | 87.0   | 96.7   | 31.1  | 67.4   | 56.0  |
| 4    | o3-mini             | 🥈79.9 | 79.5   | 🥈92.8 | 97.0   | 31.1  | 69.7   | 50.5  |
| 5    | Claude-3.7-Sonnet   | 70.0   | 88.1   | 85.1   | 57.6   | 30.4  | 75.3   | 61.8  |
| 6    | Qwen3-235B-A22B     | 63.2   | 95.3   | 87.1   | 94.1   | 30.9  | 66.7   | 55.5  |
| 7    | o4-mini             | 74.2   | 84.6   | 81.0   | 🥇98.1 | 29.0  | 🥈81.1 | 56.3  |
| 8    | GPT-4.1             | 🥉76.8 | 80.2   | 87.6   | 63.5   | 29.4  | 75.4   | 59.8  |
| 9    | DeepSeek-R1         | 59.9   | 90.6   | 89.2   | 95.1   | 27.3  | 69.0   | 56.5  |
| 10   | Grok-3-Mini         | 73.4   | 85.1   | 87.7   | 96.4   | 30.9  | 65.9   | 51.2  |
| 11   | GPT-4o              | 66.4   | 87.7   | 82.0   | 57.7   | 30.3  | 69.3   | 60.3  |
| 12   | DeepSeek-V3         | 65.2   | 92.8   | 82.1   | 57.7   | 30.9  | 68.6   | 51.5  |
| 13   | Gemini-2.5-Pro      | 61.1   | 78.7   | 🥉90.3 | 🥉97.2 | 🥉31.5| 32.6   | 54.5  |
| 14   | Qwen3-30B-A3B       | 69.0   | 81.4   | 80.1   | 92.3   | 🥈31.6| 64.9   | 54.0  |
| 15   | Qwen3-32B           | 63.1   | 90.2   | 86.0   | 94.0   | 30.4  | 65.2   | 53.5  |
| 16   | Claude-3.5-Sonnet   | 59.5   | 🥈96.5 | 81.7   | 60.1   | 30.0  | 73.2   | 47.7  |
| 17   | LLaMA-3.3-70B       | 40.7   | 🥉96.0 | 70.0   | 47.2   | 30.7  | 66.7   | 🥉62.3 |
| 18   | GPT-4-turbo         | 59.5   | 90.0   | 80.1   | 53.6   | 29.7  | 67.7   | 59.8  |
| 19   | Qwen2.5-72B         | 63.8   | 86.5   | 72.5   | 48.2   | 31.3  | 64.8   | 52.3  |
| 20   | Qwen2.5-Coder-32B   | 62.5   | 86.8   | 74.6   | 56.2   | 31.1  | 65.0   | 51.7  |
| 21   | Gemma-3-27B         | 51.3   | 83.0   | 65.9   | 41.6   | 🥇31.7| 64.7   | 62.0  |
| 22   | Claude-3.5-Haiku    | 50.9   | 85.2   | 75.0   | 46.1   | 30.6  | 44.6   | 61.2  |
| 23   | LLaMA-3.1-70B       | 48.7   | 74.5   | 67.7   | 41.5   | 30.2  | 66.3   | 57.2  |
| 24   | LLaMA-4-Scout       | 51.2   | 74.4   | 64.4   | 48.4   | 30.1  | 68.7   | 49.0  |
| 25   | GPT-3.5-turbo       | 50.6   | 71.2   | 66.5   | 34.8   | 30.4  | 67.5   | 45.8  |
| 26   | LLaMA-3.1-8B        | 31.8   | 64.2   | 49.6   | 28.8   | 30.2  | 46.0   | 54.5  |

🏅 **Legend**:  
- 🥇 = 1st place per column  
- 🥈 = 2nd place per column  
- 🥉 = 3rd place per column

---

To specify the model, task, and the parameter settings that you want, you need to specify and change the requirements in `configs/configs.yaml`, the example is already written in there!

After configruation, you simply only need to do:
```python
uv run scripts/run_experiment.py
```

When you finish testing you can evaluate by doing:
```python
uv run scripts/run_evaluation.py
```

**Note**: We provided the uv.lock and pyproject.toml, so you can easily reproduce the environment and run the experiments.

---
## Dataset
## 📂 Datasets by Task

Each TREAT task is backed by curated datasets hosted on the Hugging Face Hub: https://huggingface.co/Code-TREAT.  
You can load them in two ways:

1. **Via the `datasets` library**:
```python
from datasets import load_dataset
ds = load_dataset("Code-TREAT/<dataset_name>")
```

2.	By downloading raw JSON directly from the raw/ directory of each dataset repo, and put them in the specific positions.
 
⚡ For easier reproducibility of our testing questions, we strongly recommend using the **lite versions**.

| Task | Dataset | Reproducible & Lightweight | Description |
|------|---------|-----------------------------|-------------|
| **Code Generation (CG)** | [Code-TREAT/code_generation](https://huggingface.co/datasets/Code-TREAT/code_generation) | [Code-TREAT/code_generation_lite](https://huggingface.co/datasets/Code-TREAT/code_generation_lite) | From our self-collected HackerRank and GeeksforGeeks competitive programming dataset. Only GeeksforGeeks is stored here; HackerRank is included in Code-TREAT/code_translation as it is also part of the Translation task. |
| **Code Summarization (CS)** | [Code-TREAT/code_summarization](https://huggingface.co/datasets/Code-TREAT/code_summarization) | [Code-TREAT/code_summarization_lite](https://huggingface.co/datasets/Code-TREAT/code_summarization_lite) | From self-collected GitHub projects created since 2023 and crawled in 2025. Contains crucial function–docstring pairs. |
| **Code Translation (CT)** | [Code-TREAT/code_translation](https://huggingface.co/datasets/Code-TREAT/code_translation) | [Code-TREAT/code_translation_lite](https://huggingface.co/datasets/Code-TREAT/code_translation_lite) | From our HackerRank and GeeksforGeeks datasets. Includes PolyHumanEval implicitly via TREAT’s `benchmark_modules`. Related Paper: [Unraveling the Potential of LLMs in Code Translation](https://arxiv.org/abs/2410.09812). |
| **Code Reasoning (CR)** | [Code-TREAT/code_reasoning](https://huggingface.co/datasets/Code-TREAT/code_reasoning) | [Code-TREAT/code_reasoning_lite](https://huggingface.co/datasets/Code-TREAT/code_reasoning_lite) | Extended from HackerRank and GeeksforGeeks datasets by masking inputs/outputs, designed to test LLM reasoning via prediction accuracy. |
| **Code Review Generation (CRv)** | [Code-TREAT/code_review](https://huggingface.co/datasets/Code-TREAT/code_review) | [Code-TREAT/code_review_lite](https://huggingface.co/datasets/Code-TREAT/code_review_lite) | From self-collected GitHub projects created since 2023 and crawled in 2025. Contains diff–review pairs. |
| **Test Generation (TG)** | [Code-TREAT/unit_test_generation](https://huggingface.co/datasets/Code-TREAT/unit_test_generation) | Supplement: `_supp` version adds **branch coverage info** | From [Code-Aware Prompting](https://arxiv.org/abs/2402.00097). The original dataset is available on [Figshare](https://figshare.com/articles/dataset/SymPrompt_Focal_Method_Benchmark_for_Unit_Test_Generation/25277314?file=44661979). |
| **Vulnerability Detection (VD)** | [colin/PrimeVul](https://huggingface.co/datasets/colin/PrimeVul) | reproduce through TREAT's `replication_manifest_json/replicate_vd.json` | From [PrimeVul](https://github.com/DLVulDet/PrimeVul). We use a mirrored version on HF for convenience. Includes both single-function and paired-function variants. |

## Citation

If you use TREAT in your research, please cite:

