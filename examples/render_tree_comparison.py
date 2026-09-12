"""Render actual patch-22/current table bead centres in matching 3D views.

    python examples/render_tree_comparison.py --baseline ../support_beed_patch22

Each checkout generates coordinates in an isolated subprocess. The figure uses
identical axes, camera angles and marker sizes; it is not a timing comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings


def worker(repo):
    sys.path.insert(0, str(repo / "src"))
    from compare_tree_support import bead_records, make_model
    from pellet_support.params import SupportGenParams
    from pellet_support.pipeline import generate_support, make_params

    contact, body = make_params(nozzle_diameter_mm=0.4, bead_diameter_mm=0.2)
    gen = SupportGenParams(nozzle_diameter_mm=0.4, tree_enabled=True,
                           layer_height_mm=contact.layer_height_mm(),
                           detection_layer_height_mm=0.25, xy_clearance_mm=0.3,
                           contact_z_gap_mm=0.2, min_island_area_mm2=0.5)
    with warnings.catch_warnings(record=True), contextlib.redirect_stdout(io.StringIO()):
        result = generate_support(make_model("table"), gen, contact, body,
                                  detail=0, verbose=False)
    print(json.dumps({"beads": bead_records(result.plan, gen.layer_height_mm),
                      "tree_stats": getattr(result.plan, "tree_stats", {})}))


def generate(repo):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--repo", str(repo)]
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True,
                            timeout=180)
    if result.returncode:
        raise SystemExit(f"Generation failed in {repo}:\n{result.stderr}")
    return json.loads(result.stdout)


def render(before, after, output, baseline_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from compare_tree_support import make_model

    model = make_model("table")
    datasets = [np.asarray(before["beads"]), np.asarray(after["beads"])]
    if any(len(data) == 0 for data in datasets):
        raise SystemExit("Both checkouts must generate support beads to make this comparison.")
    reduction = 100 * (1 - len(datasets[1]) / len(datasets[0]))
    fig = plt.figure(figsize=(13.6, 7.6), dpi=180, facecolor="#f8fafc")
    fig.suptitle("Tree support bead comparison", fontsize=22, fontweight="bold",
                 color="#15263d", y=0.96)
    fig.text(0.5, 0.907, "Actual bead centres  |  Table 12 × 8 × 7 mm  |  Nozzle 0.4 mm / bead 0.2 mm",
             ha="center", fontsize=11.5, color="#4a5d73")
    titles = [f"Before · patch-22 ({baseline_ref})\n{len(datasets[0]):,} beads",
              f"After · sparse contacts\n{len(datasets[1]):,} beads  ·  {reduction:.1f}% fewer"]
    for index, (data, title) in enumerate(zip(datasets, titles), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d", facecolor="#f8fafc")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#15263d", pad=14)
        shell = Poly3DCollection(model.triangles, facecolor="#9ca8b4", alpha=0.13,
                                  edgecolor="#67788a", linewidth=0.35)
        ax.add_collection3d(shell)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2.0, c="#1769aa",
                    alpha=0.78, linewidths=0, depthshade=False, rasterized=True)
        ax.set(xlim=(-6.7, 6.7), ylim=(-4.7, 4.7), zlim=(0, 7.6))
        ax.set_box_aspect((13.4, 9.4, 7.6))
        ax.view_init(elev=24, azim=-58)
        ax.set_xlabel("X (mm)", labelpad=4, color="#536579")
        ax.set_ylabel("Y (mm)", labelpad=4, color="#536579")
        ax.set_zlabel("Z (mm)", labelpad=3, color="#536579")
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_yticks([-4, 0, 4])
        ax.set_zticks([0, 2, 4, 6])
        ax.tick_params(labelsize=8, colors="#637389", pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#e3e8ef")
            axis._axinfo["grid"]["color"] = "#e3e8ef"
    fig.subplots_adjust(left=0.035, right=0.965, top=0.79, bottom=0.19, wspace=0.06)
    fig.text(0.5, 0.11, "Same scale, camera and point size. Blue: generated bead centres. Gray: source model.",
             ha="center", fontsize=10, color="#43566c")
    spacing = after.get("tree_stats", {}).get("contact_spacing_mm", 0.8)
    fig.text(0.5, 0.068,
             f"Detection layer 0.25 mm (explicit setting); XY clearance 0.3 mm; Z gap 0.2 mm. "
             f"After: automatic contact spacing {spacing:g} mm (4 × bead diameter).",
             ha="center", fontsize=9, color="#637389")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--baseline", type=Path, default=repo.parent / "support_beed_patch22")
    parser.add_argument("--output", type=Path, default=repo / "docs" / "tree_before_after.png")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.repo.resolve())
        return
    baseline = args.baseline.resolve()
    git = subprocess.run(["git", "-C", str(baseline), "rev-parse", "--short", "HEAD"],
                         text=True, capture_output=True, check=True)
    before = generate(baseline)
    after = generate(args.repo.resolve())
    render(before, after, args.output.resolve(), git.stdout.strip())
    print(f"Saved {args.output.resolve()} ({len(before['beads']):,} -> {len(after['beads']):,} beads)")


if __name__ == "__main__":
    main()
