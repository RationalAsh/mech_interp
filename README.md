# Mechanistic Interpretability

A step-by-step workspace for reproducing mechanistic interpretability results
and developing follow-up experiments.

## Research path

1. [`experiments/01_grokking`](experiments/01_grokking) — reproduce the original
   grokking results.
2. [`experiments/02_progress_measures`](experiments/02_progress_measures) —
   reproduce the progress-measures analysis of grokking.
3. [`experiments/03_explorations`](experiments/03_explorations) — extensions and
   original experiments informed by the reproductions.

Each stage contains its own configs, notebooks, scripts, and notes. Code that is
useful in more than one stage belongs in `src/mech_interp`.

## Setup

```bash
uv sync
uv run python -c "import mech_interp; mech_interp.main()"
```

Add runtime and development dependencies with:

```bash
uv add <package>
uv add --dev <package>
```

## Layout

```text
experiments/       Numbered paper reproductions and original explorations
src/mech_interp/   Reusable data, model, training, analysis, and plotting code
tests/             Tests for reusable code
data/              Local raw and processed datasets (ignored by Git)
artifacts/         Checkpoints, logs, and figures (ignored by Git)
papers/            Paper references and reading notes
```

Keep small, durable summaries and experiment metadata in Git. Keep generated
datasets, checkpoints, logs, and figures under the ignored directories.
