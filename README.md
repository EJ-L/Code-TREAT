# TREAT: Code LLMs Trustworthiness / Reliability Evaluation And Testing 

## Overview
Project website & live leaderboard: [here](https://code-treat.vercel.app)

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

Based on this framework, we evaluated over **26 state-of-the-art models** and uncovered key insights:

- 📌 Current models show substantial **performance variation across programming tasks**.  
- 📌 **Multi-modal language models** demonstrate limitations in UI code generation and modification.  
- 📌 Existing models exhibit **severe robustness issues** on coding tasks.  
- 📌 Our **multi-prompt evaluation method** mitigates prompt bias and yields more reliable results.  

---
## Prompt Templates

All prompt templates used in our experiments are available at:

➡️ [`templates/prompts.json`](./templates/prompts.json)

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

## Leaderboard: Code Summarization (CS) by Language (Score)

| Rank | Model | Overall | Python | Java | C | C++ | C# | Ruby | JavaScript | TypeScript | PHP | Go |
|------|------------------------------|---------|--------|--------|------|-------|------|-------|------------|------------|------|------|
| 1 | Claude-3.5-Sonnet-20241022 | 66.1 | 67.6 | 68.6 | 68.2 | 62.9 | 65.3 | 65.9 | 63.7 | 67.1 | 67.4 | 64.6 |
| 2 | Llama-3.3-70B-Instruct | 65.9 | 68.9 | 69.3 | 67.0 | 63.4 | 64.9 | 64.6 | 62.5 | 66.4 | 67.8 | 64.4 |
| 3 | Claude-Sonnet-4 | 65.8 | 67.7 | 69.6 | 67.7 | 63.5 | 64.8 | 65.1 | 62.8 | 66.3 | 66.9 | 64.1 |
| 4 | GPT-5 | 65.7 | 69.4 | 68.4 | 68.4 | 63.1 | 66.1 | 62.9 | 63.0 | 67.0 | 67.9 | 66.7 |
| 5 | Qwen3-235B-A22B | 65.1 | 68.1 | 65.1 | 66.8 | 62.2 | 65.3 | 62.8 | 62.3 | 64.7 | 70.7 | 67.1 |
| 6 | DeepSeek-V3 | 64.3 | 67.2 | 66.3 | 67.0 | 61.9 | 63.9 | 63.5 | 61.2 | 64.1 | 65.0 | 63.0 |
| 7 | DeepSeek-R1 (0528) | 63.8 | 67.3 | 66.3 | 66.4 | 61.3 | 63.1 | 62.0 | 61.0 | 63.9 | 64.3 | 62.1 |
| 8 | DeepSeek-R1 | 63.8 | 67.2 | 66.4 | 66.4 | 61.4 | 63.2 | 62.0 | 61.0 | 63.9 | 64.2 | 62.1 |
| 9 | Claude-3.7-Sonnet | 63.7 | 66.7 | 65.8 | 65.1 | 60.9 | 62.3 | 64.3 | 60.5 | 63.8 | 65.2 | 62.4 |
| 10 | GPT-4-turbo-2024-04-09 | 63.2 | 66.3 | 64.2 | 66.2 | 61.3 | 63.6 | 63.2 | 60.9 | 63.1 | 63.6 | 63.7 |
| 11 | Qwen3-32B | 63.1 | 68.0 | 63.4 | 67.1 | 61.1 | 63.9 | 62.0 | 60.9 | 63.4 | 64.7 | 61.7 |
| 12 | GPT-4o-2024-11-20 | 62.8 | 67.2 | 63.1 | 65.5 | 60.8 | 63.2 | 63.4 | 59.7 | 62.9 | 64.4 | 62.6 |
| 13 | Qwen2.5-72B-Instruct | 62.6 | 66.0 | 64.5 | 64.8 | 59.8 | 62.3 | 62.0 | 59.4 | 62.5 | 63.3 | 61.6 |
| 14 | Qwen2.5-Coder-32B-Instruct | 62.6 | 65.8 | 66.1 | 64.8 | 59.6 | 61.2 | 60.9 | 61.0 | 62.3 | 63.8 | 62.3 |
| 15 | Grok-3-Mini (High) | 62.5 | 64.6 | 65.9 | 65.4 | 60.3 | 60.8 | 60.9 | 59.8 | 62.8 | 63.7 | 61.2 |
| 16 | Claude-3.5-Haiku-20241022 | 61.6 | 64.4 | 64.7 | 64.8 | 59.2 | 60.0 | 55.8 | 59.7 | 62.2 | 63.1 | 62.2 |
| 17 | Gemma-3-27B-Instruct | 61.3 | 64.8 | 64.1 | 62.2 | 57.5 | 60.6 | 61.3 | 58.2 | 61.9 | 62.9 | 60.0 |
| 18 | o4-mini (Med) | 61.1 | 66.1 | 62.8 | 63.3 | 59.3 | 60.6 | 57.9 | 59.0 | 62.6 | 62.3 | 62.8 |
| 19 | o3-mini (Med) | 60.4 | 24.5 | 60.3 | 64.3 | 59.3 | 62.0 | 60.8 | 59.0 | 61.5 | 63.6 | 62.1 |
| 20 | Gemini-2.5-Pro-Preview-05-06 | 60.3 | 63.3 | 62.5 | 62.6 | 57.4 | 59.4 | 58.6 | 58.0 | 61.1 | 62.3 | 60.2 |
| 21 | GPT-4.1-2025-04-14 | 60.0 | 62.4 | 62.0 | 62.7 | 57.9 | 58.9 | 58.8 | 57.6 | 60.7 | 62.2 | 61.1 |
| 22 | Qwen3-30B-A3B | 59.7 | 63.5 | 59.6 | 62.4 | 57.1 | 60.6 | 59.6 | 57.2 | 60.5 | 61.2 | 59.3 |
| 23 | Llama-3.1-70B-Instruct | 58.6 | 58.4 | 62.2 | 60.2 | 56.5 | 59.0 | 55.2 | 56.4 | 59.3 | 61.9 | 56.8 |
| 24 | Llama-4-Scout-17B-16E-Instruct | 58.4 | 59.3 | 63.3 | 58.9 | 54.6 | 57.3 | 57.0 | 55.7 | 58.6 | 60.8 | 58.2 |
| 25 | GPT-3.5-turbo-0125 | 56.3 | 58.8 | 59.2 | 58.1 | 53.1 | 55.6 | 56.0 | 52.7 | 55.9 | 58.4 | 55.1 |
| 26 | Llama-3.1-8B-Instruct | 54.3 | 56.2 | 57.3 | 53.9 | 51.6 | 53.4 | 54.5 | 51.3 | 54.7 | 56.1 | 54.1 |

---

## Leaderboard: Code Review Generation (CRv) by Language (Score)

| Rank | Model | Overall | Python | Java | C | C++ | C# | Ruby | JavaScript | TypeScript | PHP | Go |
|------|------------------------------|---------|--------|--------|------|------|------|------|------------|------------|------|------|
| 1 | Grok-3-Mini (High) | 35.3 | 35.5 | 34.7 | 33.0 | 35.3 | 35.3 | 33.9 | 37.2 | 37.0 | 34.0 | 37.2 |
| 2 | Claude-Sonnet-4 | 35.0 | 34.8 | 33.9 | 33.0 | 34.6 | 34.8 | 34.5 | 38.1 | 35.4 | 34.2 | 36.7 |
| 3 | DeepSeek-R1 (0528) | 34.9 | 34.8 | 34.4 | 32.5 | 35.1 | 35.7 | 33.8 | 36.7 | 36.5 | 33.5 | 36.6 |
| 4 | Claude-3.7-Sonnet | 34.8 | 34.9 | 34.5 | 32.9 | 34.8 | 35.7 | 33.0 | 36.4 | 36.1 | 34.0 | 36.1 |
| 5 | Gemini-2.5-Pro-05-06 | 34.8 | 35.2 | 34.9 | 31.8 | 35.5 | 33.9 | 35.0 | 37.4 | 36.1 | 33.0 | 35.3 |
| 6 | Gemma-3-27B-Instruct | 34.7 | 34.8 | 33.8 | 32.3 | 34.4 | 35.1 | 34.8 | 37.1 | 35.3 | 33.9 | 35.5 |
| 7 | Claude-3.5-Sonnet-20241022 | 34.6 | 33.9 | 32.8 | 31.7 | 34.2 | 34.3 | 33.5 | 36.9 | 35.1 | 33.6 | 36.4 |
| 8 | Qwen3-30B-A3B | 34.6 | 35.3 | 33.3 | 32.8 | 35.4 | 34.2 | 33.7 | 35.7 | 34.8 | 33.6 | 37.5 |
| 9 | Qwen3-235B-A22B | 34.5 | 34.9 | 33.0 | 32.1 | 34.6 | 34.5 | 33.7 | 37.7 | 35.4 | 33.4 | 36.7 |
| 10 | o3-mini (Med) | 34.5 | 33.1 | 32.8 | 30.9 | 34.8 | 34.4 | 34.2 | 36.8 | 34.7 | 33.6 | 36.0 |
| 11 | GPT-4.1-2025-04-14 | 34.4 | 34.3 | 33.5 | 31.9 | 33.6 | 33.5 | 33.3 | 36.7 | 36.8 | 33.3 | 36.8 |
| 12 | Qwen2.5-72B-Instruct | 34.4 | 32.7 | 33.5 | 31.4 | 34.3 | 34.7 | 33.5 | 36.3 | 35.2 | 33.1 | 35.9 |
| 13 | Qwen3-32B | 34.2 | 34.8 | 33.9 | 32.5 | 34.2 | 34.0 | 33.5 | 36.0 | 34.8 | 33.0 | 35.7 |
| 14 | Claude-3.5-Haiku-20241022 | 34.1 | 34.4 | 33.5 | 32.2 | 34.4 | 33.9 | 33.7 | 35.0 | 34.8 | 33.8 | 35.4 |
| 15 | Llama-4-Scout-17B-16E-Instruct | 34.1 | 34.4 | 34.1 | 32.4 | 34.3 | 34.5 | 33.4 | 36.3 | 33.4 | 32.8 | 35.3 |
| 16 | Qwen2.5-Coder-32B-Instruct | 34.1 | 32.5 | 33.0 | 30.9 | 34.2 | 34.1 | 33.4 | 36.4 | 35.5 | 32.9 | 34.8 |
| 17 | Llama-3.3-70B-Instruct | 33.9 | 32.1 | 32.0 | 31.0 | 33.3 | 34.6 | 33.7 | 36.0 | 34.6 | 32.5 | 35.1 |
| 18 | GPT-4-turbo-2024-04-09 | 33.8 | 33.6 | 33.5 | 31.4 | 33.4 | 33.7 | 33.1 | 35.8 | 34.5 | 33.6 | 35.2 |
| 19 | GPT-4o-2024-11-20 | 33.8 | 30.7 | 31.6 | 30.2 | 33.9 | 33.4 | 33.2 | 36.2 | 34.1 | 32.1 | 34.4 |
| 20 | o4-mini (Med) | 33.5 | 32.8 | 32.8 | 31.6 | 33.0 | 33.8 | 32.4 | 36.4 | 35.6 | 31.8 | 34.7 |
| 21 | Llama-3.1-70B-Instruct | 33.3 | 32.6 | 30.1 | 30.9 | 33.6 | 33.6 | 32.6 | 34.9 | 33.6 | 32.5 | 34.8 |
| 22 | DeepSeek-R1 | 33.3 | 33.9 | 32.4 | 31.3 | 32.9 | 33.1 | 31.6 | 35.2 | 34.5 | 33.8 | 35.1 |
| 23 | GPT-5 | 33.1 | 33.0 | 32.3 | 30.6 | 32.0 | 33.9 | 32.4 | 35.5 | 35.0 | 32.7 | 34.7 |
| 24 | Llama-3.1-8B-Instruct | 32.6 | 31.3 | 32.8 | 30.0 | 32.6 | 32.0 | 32.6 | 33.6 | 33.2 | 31.5 | 34.4 |
| 25 | GPT-3.5-turbo-0125 | 31.3 | 30.3 | 30.0 | 27.1 | 32.1 | 31.4 | 30.3 | 31.3 | 31.6 | 30.3 | 33.5 |
| 26 | DeepSeek-V3 | 25.7 | 25.2 | 25.0 | 23.4 | 25.4 | 25.5 | 24.8 | 27.4 | 26.6 | 25.3 | 26.5 |

---

🏅 **Legend**:  
- 🥇 = 1st place per column  
- 🥈 = 2nd place per column  
- 🥉 = 3rd place per column

---

You control experiments through `configs/configs.yaml`. Each task stanza accepts the same set of knobs (models, datasets, sampling, manifests, etc.) while the new `reproduce` flag (default `true`) makes runners stick to the `_lite` datasets we used for our public results; flip it to `false` when you want to exercise the full corpora once you have them locally.

Here is a minimal example that runs three tasks and saves per-dataset manifests under `results/<task>/<dataset>_sample_manifest.json`:

```yaml
model_specification:
  models: ["glm-4-flash"]
  temperature: 0.8
  top_k: 40
  top_p: 0.95

tasks:
  - task: code_translation
    enabled: true
    parameters:
      datasets: [{hackerrank: "java->python"}]
      reproduce: true          # use the curated `_lite` split
      sampling_mode: "random"  # or "manifest" to replay an existing file
      sample_size: 3
      sampling_seed: 42
      sampling_manifest_path: ""  # leave empty to use the default results/<task>/<dataset> path
      template_categories: ["direct"]
      save_dir: "results"
      parallel_requests: 2

  - task: vulnerability_detection
    enabled: true
    parameters:
      datasets: ["primevul", "primevul_pair"]
      reproduce: true
      sampling_mode: "random"
      sample_size: 2
      sampling_manifest_path: ""  # default path per dataset

  - task: code_generation
    enabled: true
    parameters:
      datasets: [{hackerrank: "java"}]
      reproduce: false          # full dataset (requires HF access or local copy)
      sampling_mode: "manifest"
      sampling_manifest_path: "replication_manifest_json/code_generation_manifest.json"
```

With the config in place, simply run:
```python
uv run scripts/run_experiment.py
```

Every run writes a sampling manifest for each dataset under `results/<task>/` (unless you override `sampling_manifest_path`). Subsequent runs in `manifest` mode read the same file and now preserve the entries for previously sampled datasets instead of overwriting them, so you can accumulate several slices in a single manifest.

When you want to score completed generations, point the evaluation driver at the same config (so it knows which datasets/models to look for) and override judge settings on the CLI as needed:

```bash
uv run scripts/run_evaluation.py \
  --tasks code_summarization code_translation \
  --judge-model gpt-4o-2024-11-20 \
  --max-workers 4
```

`run_evaluation.py` walks each enabled task, pulls predictions from `results/<task>/<dataset>/predictions`, populates `/parsed/` if required, and drops evaluation files in `/evaluations/`. The CLI flags simply override what’s in `configs/configs.yaml` (e.g., pick a different judge model or limit tasks to re-evaluate).

The mapping between config and filesystem is one-to-one:

| `task` | Example `datasets` entry | Predictions expected in… | Evaluations written to… |
|--------|--------------------------|---------------------------|--------------------------|
| `code_summarization` | `{'github_2023': 'python'}` | `results/code_summarization/github_2023/predictions/` | `results/code_summarization/github_2023/evaluations/` |
| `code_translation` | `{'hackerrank': 'java->python'}` | `results/code_translation/hackerrank/predictions/` | `results/code_translation/hackerrank/evaluations/` |
| `code_generation` | `{'hackerrank': 'java'}` | `results/code_generation/hackerrank/predictions/` | `results/code_generation/hackerrank/evaluations/` |
| `code_review_generation` | `{'github_2023': 'python'}` | `results/code_review_generation/github_2023/predictions/` | `results/code_review_generation/github_2023/evaluations/` |
| `unit_test_generation` | `{'symprompt': 'python'}` | `results/unit_test_generation/symprompt/predictions/` | `results/unit_test_generation/symprompt/evaluations/` |
| `vulnerability_detection` | `'primevul'` or `'primevul_pair'` | `results/vulnerability_detection/<dataset>/predictions/` | `results/vulnerability_detection/<dataset>/evaluations/` |

As long as the tasks/datasets listed in `configs/configs.yaml` match the directories populated during generation, the evaluator will locate them automatically. For example, predictions stored in `results/code_summarization/github_2023/predictions/GLM-4-Flash.jsonl` produce an evaluation file at `results/code_summarization/github_2023/evaluations/GLM-4-Flash.jsonl` (with judge decisions cached alongside).

If you do want a separate evaluation configuration (e.g., to change judges without touching your experiment settings), copy `configs/configs.yaml`, edit the `model_specification`/`tasks` blocks as needed, and pass it via `--config path/to/evaluation.yaml`.

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
| **Vulnerability Detection (VD)** | [Code-TREAT/PrimeVul_original](https://huggingface.co/datasets/Code-TREAT/PrimeVul_original), [Code-TREAT/PrimeVul-Paired_original](https://huggingface.co/datasets/Code-TREAT/PrimeVul-Paired_original) | [Code-TREAT/PrimeVul_original_lite](https://huggingface.co/datasets/Code-TREAT/PrimeVul_original_lite), [Code-TREAT/PrimeVul-Paired_original_lite](https://huggingface.co/datasets/Code-TREAT/PrimeVul-Paired_original_lite) | Mirrors of the PrimeVul single-function and paired variants we curated for reproducible evaluation. Set `reproduce: true` to target the `_lite` splits shipped with the repo. |

## Citation

If you use TREAT in your research, please cite:
