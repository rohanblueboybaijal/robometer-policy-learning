"""Generate local PNG/PDF curves and honest, paired comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import write_json


def read_curve(path):
    with Path(path).open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def paired_difference(left, right):
    """Bootstrap over common held-out initial states, NOT over training seeds."""
    left_by_state = {r["init_state_id"]: float(r["success"]) for r in left}
    right_by_state = {e["init_state_id"]: float(e["success"]) for e in right}
    if left_by_state.keys() != right_by_state.keys() or not left_by_state:
        return {"error": "Held-out initial states are not identical"}
    delta = np.array([left_by_state[k] - right_by_state[k] for k in sorted(left_by_state)])
    rng = np.random.default_rng(0)
    draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "paired_states": len(delta),
        "uncertainty": "Reset-state bootstrap for a single training seed; not seed-level uncertainty",
    }


def plot_iterations(directory, runs, groups):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
    data = {}
    for column, group in enumerate(groups):
        ax = axes[0, column]
        selected = [run for run in runs if run["config"]["comparison"] == group]
        counts = {run["config"]["num_envs"] for run in selected}
        for run in selected:
            cfg, curve = run["config"], run["curve"]
            xs = [point["env_steps"] / cfg["num_envs"] for point in curve]
            label = cfg["name"]
            if run["status"]["status"] != "complete":
                label += " (incomplete)"
            line = ax.plot(xs, [point["success_rate"] for point in curve], marker=".", label=label)[0]
            ax.fill_between(
                xs,
                [point["ci_low"] for point in curve],
                [point["ci_high"] for point in curve],
                color=line.get_color(),
                alpha=0.13,
            )
            data[cfg["name"]] = [dict(point, env_iterations=x) for point, x in zip(curve, xs, strict=False)]
        if len(counts) == 1:
            count = next(iter(counts))
            secondary = ax.secondary_xaxis("top", functions=(lambda x, n=count: x * n, lambda x, n=count: x / n))
            secondary.set_xlabel("Environment transitions (all parallel environments)")
            xlabel = f"Environment iterations (one step in each of {count} environments)"
        else:
            xlabel = "Environment iterations (environment count varies by run)"
        cfg = selected[0]["config"]
        ax.set(
            title=f"{cfg['suite']}: {cfg['task'].replace('_', ' ')}",
            xlabel=xlabel,
            ylabel="Simulator success rate",
            ylim=(-0.02, 1.02),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("SAC reward comparison — single training seed")
    fig.tight_layout()
    fig.savefig(directory / "success_rate_vs_iterations.png", dpi=180)
    fig.savefig(directory / "success_rate_vs_iterations.pdf")
    plt.close(fig)
    write_json(directory / "iteration_curves.json", data)


def generate_report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    runs = []
    for path in sorted(directory.glob("*/config.json")):
        folder = path.parent
        curve_path = folder / "evaluation.csv"
        if not curve_path.exists():
            continue
        curves = read_curve(curve_path)
        if not curves:
            continue
        cfg = json.loads(path.read_text())
        status_file = folder / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {"status": "unknown"}
        episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        runs.append({"config": cfg, "status": status, "curve": curves, "episodes": episodes})
    if not runs:
        write_json(
            directory / "summary.json",
            {
                "status": "no_measurements",
                "message": "No LIBERO experiment curves exist. No performance claim can be made.",
            },
        )
        return
    groups = sorted({r["config"]["comparison"] for r in runs})
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
    comparisons = []
    for column, group in enumerate(groups):
        ax = axes[0, column]
        selected = [r for r in runs if r["config"]["comparison"] == group]
        baseline = next((r for r in selected if r["config"]["reward"] == "sparse"), None)
        for run in selected:
            curve = run["curve"]
            xs = [p["env_steps"] for p in curve]
            ys = [p["success_rate"] for p in curve]
            label = run["config"]["name"]
            if run["status"]["status"] != "complete":
                label += " (incomplete)"
            line = ax.plot(xs, ys, marker=".", label=label)[0]
            ax.fill_between(
                xs, [p["ci_low"] for p in curve], [p["ci_high"] for p in curve], color=line.get_color(), alpha=0.13
            )
            if baseline and run is not baseline:
                cfg, gt_cfg = run["config"], baseline["config"]
                matching = [
                    "task",
                    "seed",
                    "num_envs",
                    "steps",
                    "horizon",
                    "train_init_states",
                    "eval_episodes",
                    "eval_every",
                    "initial_actor",
                    "sac",
                    "actor_hidden",
                    "critic_hidden",
                    "dino_checkpoint",
                    "updates_per_transition",
                    "sparse_offset",
                    "buffer_capacity",
                ]
                mismatches = [k for k in matching if cfg[k] != gt_cfg[k]]
                entry = {
                    "run": cfg["name"],
                    "baseline": gt_cfg["name"],
                    "mismatches": mismatches,
                    "reward_formula": cfg["reward_formula"],
                    "ood_claim": cfg["ood_claim"],
                }
                complete = run["status"]["status"] == baseline["status"]["status"] == "complete"
                if not complete or mismatches:
                    entry["conclusion"] = "incomplete_or_unmatched_comparison"
                else:
                    step = curve[-1]["env_steps"]
                    gt_step = baseline["curve"][-1]["env_steps"]
                    model_episodes = [e for e in run["episodes"] if e["phase"] == "eval" and e["env_steps"] == step]
                    sparse_episodes = [
                        e for e in baseline["episodes"] if e["phase"] == "eval" and e["env_steps"] == gt_step
                    ]
                    entry["final"] = paired_difference(model_episodes, sparse_episodes)
                    entry["initial_success_difference"] = ys[0] - baseline["curve"][0]["success_rate"]
                    entry["sparse_improvement"] = (
                        baseline["curve"][-1]["success_rate"] - baseline["curve"][0]["success_rate"]
                    )
                    gt_points = {p["env_steps"]: p["success_rate"] for p in baseline["curve"]}
                    if set(xs) == set(gt_points):
                        differences = [y - gt_points[x] for x, y in zip(xs, ys, strict=False)]
                        entry["normalized_success_auc_difference"] = float(np.trapz(differences, xs) / xs[-1])
                    if step != gt_step:
                        entry["conclusion"] = "unmatched_final_budget"
                    elif entry["initial_success_difference"] != 0:
                        entry["conclusion"] = "different_initial_policies_or_evaluations"
                    elif "error" in entry["final"]:
                        entry["conclusion"] = "unmatched_evaluation_states"
                    elif entry["final"]["difference"] < 0 and entry["sparse_improvement"] > 0:
                        entry["conclusion"] = "robometer_underperforms_in_this_seed"
                    else:
                        entry["conclusion"] = "no_evidence_of_reward_harm_in_this_seed"
                comparisons.append(entry)
        ax.set(
            title=f"{group}: simulator success",
            xlabel="Aggregate physical environment steps",
            ylabel="Success rate",
            ylim=(-0.02, 1.02),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("SAC with frozen Robometer vs sparse simulator reward (single seed)")
    fig.tight_layout()
    fig.savefig(directory / "performance.png", dpi=180)
    fig.savefig(directory / "performance.pdf")
    plt.close(fig)
    plot_iterations(directory, runs, groups)
    write_json(
        directory / "summary.json",
        {
            "status": "measured",
            "comparisons": comparisons,
            "limitations": [
                "Single training seed",
                "Curves show Wilson intervals over reset states, not training seeds",
                "Released general checkpoint is not the paper's LIBERO-only H2 ablation checkpoint",
                "Wrist arm tests an explicit reward-input shift; it is not proof of natural task OOD failure",
            ],
        },
    )
    lines = [
        "# Local reward audit results",
        "",
        "Evaluation uses simulator success on held-out initial states.",
        "Shaded bands show reset-state Wilson intervals; this is a single training seed.",
        "",
        "| Run | Sparse comparator | Outcome |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c['run']} | {c['baseline']} | {c['conclusion']} |" for c in comparisons]
    lines += [
        "",
        "See summary.json for paired differences, initial-policy checks, budgets, and uncertainty.",
        "The sanity run follows the LIBERO SAC protocol but uses the publicly released general checkpoint.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    generate_report(args.directory)


if __name__ == "__main__":
    main()
