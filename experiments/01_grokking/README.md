# 01 — Grokking reproduction

Reproduce the core results from *Grokking: Generalization Beyond Overfitting on
Small Algorithmic Datasets*.

Suggested milestones:

1. Implement modular-arithmetic datasets and deterministic train/test splits.
2. Train the baseline transformer and reproduce delayed generalization.
3. Sweep training-set size, weight decay, and optimization settings.
4. Reproduce the main training/test loss and accuracy figures.
5. Record deviations from the paper and exact reproduction settings in `notes/`.

- `configs/` — versioned experiment parameters.
- `notebooks/` — exploratory analysis only.
- `scripts/` — reproducible training and evaluation entry points.
- `notes/` — results, decisions, and paper-reading notes.
