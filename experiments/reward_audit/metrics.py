from __future__ import annotations

import math

import numpy as np


def wilson(successes, trials):
    if trials == 0:
        return None, None
    p, z = successes / trials, 1.959963984540054
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - radius, center + radius


def binary_auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    diffs = positive[:, None] - negative[None, :]
    return float(((diffs > 0) + 0.5 * (diffs == 0)).mean())


def alignment(episodes):
    labels = np.array([e["success"] for e in episodes], dtype=bool)
    progress = np.array([e["final_progress"] for e in episodes], dtype=float)
    probs = np.array([e["final_success_probability"] for e in episodes], dtype=float)
    predicted = probs >= 0.65
    return {
        "n": len(episodes),
        "progress_auc": binary_auc(progress, labels),
        "success_brier": float(np.mean((probs - labels) ** 2)),
        "success_progress_mean": float(progress[labels].mean()) if labels.any() else None,
        "failure_progress_mean": float(progress[~labels].mean()) if (~labels).any() else None,
        "false_positive_rate": float(predicted[~labels].mean()) if (~labels).any() else None,
        "false_negative_rate": float((~predicted[labels]).mean()) if labels.any() else None,
    }


def training_reward(kind, formula, sparse, progress, previous):
    """Keep simulator labels out of the model-only reward formulas."""
    sparse = np.asarray(sparse, dtype=np.float32)
    if kind == "sparse":
        return sparse
    progress = np.asarray(progress, dtype=np.float32)
    if formula == "progress":
        return progress
    if formula == "progress_minus_one":
        return progress - 1
    if formula == "sparse_plus_progress":
        return sparse + progress
    if formula == "delta":
        return progress - np.asarray(previous, dtype=np.float32)
    raise ValueError(formula)


def prefix_frames(history, max_frames):
    """Evenly sample only the observed prefix, including its current endpoint."""
    n = min(max_frames, len(history))
    indices = np.linspace(0, len(history) - 1, n, dtype=int)
    return np.stack([history[i] for i in indices])
