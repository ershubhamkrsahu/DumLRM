<div align="center">

# DumLRM

**A reasoning model built from scratch — tokenizer, transformer, GRPO, distillation.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](#quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](#quick-start)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/theory-complete-brightgreen)](#status)

</div>

DumLRM is a hands-on implementation of a small reasoning language model. It starts
with the Qwen3 architecture written directly in PyTorch and follows the full path
to a reasoning model: evaluation, inference scaling, self-refinement, GRPO with
verifiable rewards, and distillation from stronger reasoning traces.

The project is notebook-driven. Every chapter is a runnable notebook that builds on
the previous one, and the shared code lives in a single, readable helper module.
No training framework hides the interesting parts.

## Highlights

- **Qwen3 0.6B from scratch** — grouped-query attention, RoPE, RMSNorm, KV cache,
  streaming generation, and `torch.compile`.
- **A real evaluation pipeline** — boxed-answer extraction, LaTeX normalization,
  sympy-based equivalence checking, and a MATH-500 harness.
- **Inference-time reasoning** — temperature and top-p sampling, self-consistency,
  sequence log-probabilities, and self-refinement.
- **GRPO for RLVR** — rollouts, rule-based rewards, advantages, clipping, KL
  control, and a format reward.
- **Reasoning distillation** — DeepSeek-R1 traces, length filtering, a
  token-level distillation loss, and a complete training loop.
- **Reproducible environment** — `uv` lockfile, CUDA 12.4 PyTorch wheels, and
  notebooks that run top to bottom.

## Chapters

Chapter 1 is the introduction. The implementation starts in Chapter 2.

| # | Notebook | What it covers |
|---|----------|----------------|
| 02 | [LLM basics](src/dumlrm/chapters/ch02_llm_basics.ipynb) | Qwen3 tokenizer and model, KV cache, streaming generation, `torch.compile` |
| 03 | [Evaluation](src/dumlrm/chapters/ch03_evaluation.ipynb) | Answer extraction, normalization, sympy grading, MATH-500 evaluation |
| 04 | [Self-consistency](src/dumlrm/chapters/ch04_self_consistency.ipynb) | Temperature scaling, top-p sampling, multiple samples, majority vote |
| 05 | [Self-refinement](src/dumlrm/chapters/ch05_self_refinement.ipynb) | Sequence log-probabilities, confidence scoring, iterative revision |
| 06 | [GRPO for RLVR](src/dumlrm/chapters/ch06_grpo_for_rlvr.ipynb) | Rollouts, rule-based rewards, advantages, policy-gradient updates |
| 07 | [Improving GRPO](src/dumlrm/chapters/ch07_improve_grpo.ipynb) | Advantage and entropy tracking, ratio clipping, KL control, format reward |
| 08 | [Distillation](src/dumlrm/chapters/ch08_distillation.ipynb) | R1 reasoning traces, dataset filtering, distillation loss, training loop |

## Datasets

The datasets are hosted on Hugging Face and are **not** tracked in Git. The loaders
inside the notebooks download them automatically on first run.

| Dataset | Contents | File |
|---------|----------|------|
| [MATH-500 test](https://huggingface.co/datasets/shubhamisok/math-500/blob/main/math500_test.json) | 500 held-out competition math problems | `math500_test.json` |
| [MATH train split](https://huggingface.co/datasets/shubhamisok/math-500/blob/main/math_full_minus_math500.json) | MATH training problems with MATH-500 removed | `math_full_minus_math500.json` |
| [DeepSeek-R1 traces](https://huggingface.co/datasets/shubhamisok/math-500/blob/main/deepseek-r1-math-train.json) | 12,000 problems with R1 thinking and final answers | `deepseek-r1-math-train.json` |

Dataset repository: [shubhamisok/math-500](https://huggingface.co/datasets/shubhamisok/math-500).

## Models

The starting checkpoint is the official Qwen3 0.6B base model. The DumLRM
checkpoints are still being trained; their links will replace the placeholders
below when training finishes.

| Model | Stage | Weights |
|-------|-------|---------|
| Qwen3-0.6B base | Starting point | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| DumLRM GRPO | Chapters 6–7 | _Coming soon_ |
| DumLRM distilled | Chapter 8 | _Coming soon_ |

<!--
When a checkpoint is uploaded, replace the matching "Coming soon" cell with:
[DumLRM-GRPO](https://huggingface.co/shubhamisok/dumlrm-grpo)
[DumLRM-Distilled](https://huggingface.co/shubhamisok/dumlrm-distilled)
-->

## Quick start

You need Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is
recommended; the model runs on CPU, but generation will be slow.

```bash
git clone https://github.com/ShubhamKumarSahu-svg/DumLRM.git
cd DumLRM
uv sync
uv run jupyter lab
```

Open the notebooks under `src/dumlrm/chapters/` and run them in order. The first
run downloads the Qwen3 weights (~1.5 GB) into `src/dumlrm/chapters/qwen3/` and the
datasets into `src/dumlrm/chapters/`. Both locations are gitignored.

> The lockfile targets CUDA 12.4 wheels. If your platform needs a different
> PyTorch build, adjust `[tool.uv.sources]` in `pyproject.toml` before syncing.

## Evaluate a results file

`evaluate_json.py` grades a JSON or JSONL file containing model outputs. It accepts
either a top-level list, a `{"records": [...]}` object, or one record per line.

```bash
cd src/dumlrm/chapters
uv run evaluate_json.py \
  --json_path math500-cuda.jsonl \
  --gtruth_answer answer \
  --generated_text generated
```

The CLI extracts the final boxed answer, normalizes it, checks mathematical
equivalence with sympy, and prints the overall accuracy.

## Project layout

```text
DumLRM/
├── src/dumlrm/
│   ├── __init__.ipynb                  # Environment check
│   └── chapters/
│       ├── ch02_llm_basics.ipynb
│       ├── ch03_evaluation.ipynb
│       ├── ch04_self_consistency.ipynb
│       ├── ch05_self_refinement.ipynb
│       ├── ch06_grpo_for_rlvr.ipynb
│       ├── ch07_improve_grpo.ipynb
│       ├── ch08_distillation.ipynb
│       ├── evaluate_json.py            # Batch grader for JSON/JSONL outputs
│       └── reasoning_utils.py          # Shared model, data, and evaluation helpers
├── my_view.md                          # Development journal
├── pyproject.toml
└── uv.lock
```

`reasoning_utils.py` is the only shared module. It contains the Qwen3 model and
tokenizer, KV cache, download and device helpers, prompt rendering, answer
extraction and grading, the sampling generators, and the GRPO rollout and scoring
functions used by the notebooks.

## Status

The theoretical portion is complete: Chapters 2 through 8 cover the full path from
the base model to distillation.

The trained models have not been uploaded yet. Checkpoints, links, and final
MATH-500 results will be added once training finishes. Progress notes and design
decisions live in [my_view.md](my_view.md).

Planned next steps:

- Finish the distillation run and evaluate the checkpoints on the full MATH-500 set.
- Upload the GRPO and distilled checkpoints to Hugging Face.
- Add the final accuracy numbers and model links to this README.

## Attribution

The chapter structure and several implementations follow *Build a Reasoning Model
(From Scratch)* by Sebastian Raschka. The upstream code is Apache-2.0 licensed and
its copyright headers are preserved in `reasoning_utils.py`.

- Book: <https://mng.bz/lZ5B>
- Upstream repository: <https://github.com/rasbt/reasoning-from-scratch>

## License

Apache License 2.0 — see [LICENSE](LICENSE). The datasets are distributed
separately on Hugging Face under the terms listed on their dataset card.
