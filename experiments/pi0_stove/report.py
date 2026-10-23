"""Local success curves for the pi0 steering runs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


def generate(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    planned = directory / "planned_runs.json"
    task = "turn_on_the_stove"
    if planned.exists():
        runs = json.loads(planned.read_text()).get("runs", [])
        if runs:
            task = runs[0].get("task", task)
    title = task.replace("_", " ")
    curves = {}
    for kind in ("sparse", "robometer"):
        path = directory / f"{kind}_seed100" / "evaluation.csv"
        rows = []
        if path.exists():
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    try:
                        rows.append({"env_steps": float(row["env_steps"]),
                                     "successes": int(float(row["successes"])),
                                     "episodes": int(float(row["episodes"])),
                                     "success_rate": float(row["success_rate"])})
                    except (TypeError, ValueError):
                        continue
        curves[kind] = rows
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for kind, color in (("sparse", "#087f8c"), ("robometer", "#d36a35")):
        points = curves[kind]
        if points:
            axis.plot([p["env_steps"] for p in points], [p["success_rate"] for p in points],
                      color=color, marker="o", ms=3, label=f"{kind.title()} reward")
    axis.set(xlabel="Training environment transitions", ylabel="Held-out simulator success rate",
             ylim=(-.03, 1.03), xlim=(0, 100000), title=f"LIBERO {title}: frozen pi0 with SAC steering")
    axis.grid(alpha=.2)
    if any(curves.values()):
        axis.legend()
    figure.tight_layout()
    for ext in ("png", "pdf"):
        figure.savefig(directory / f"success_rate_vs_transitions.{ext}", dpi=180)
    plt.close(figure)
    (directory / "report_data.json").write_text(json.dumps(curves, indent=2) + "\n")
    rows = []
    for kind in ("sparse", "robometer"):
        for point in curves[kind]:
            rows.append(f"<tr><td>{kind.title()}</td><td>{int(point['env_steps']):,}</td>"
                        f"<td>{point['successes']}/{point['episodes']}</td>"
                        f"<td>{point['success_rate']:.1%}</td></tr>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pi0 {html.escape(title)} results</title><style>body{{font:16px/1.5 system-ui;max-width:1000px;margin:auto;padding:32px;color:#17313b;background:#f5f7f6}}main{{background:white;border-radius:14px;padding:28px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#087f8c}}</style></head><body><main><h1>π₀ {html.escape(title)} reward comparison</h1><p>Frozen π₀-LIBERO policy with SAC steering. Both arms start from the same policy and use simulator success on 25 held-out reset states. Training reward is either native sparse 0/1 or Robometer progress averaged across each action chunk. No simulator success bonus enters the Robometer arm.</p><img src="success_rate_vs_transitions.png" alt="Held-out success rate over training"><p><a href="success_rate_vs_transitions.pdf">PDF curve</a> · <a href="report_data.json">Raw curve JSON</a></p><h2>Checkpoints</h2><table><thead><tr><th>Reward</th><th>Transitions</th><th>Successes</th><th>Success</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>Refresh this page for the latest saved checkpoints. Reward provenance and decoder checks are stored in each run folder.</p></main></body></html>"""
    (directory / "index.html").write_text(page)
    return directory / "index.html"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(generate(parser.parse_args().directory))
