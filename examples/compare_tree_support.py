"""Compare sparse tree support with a separate, unmodified checkout.

Run from the repository root::

    python examples/compare_tree_support.py --baseline ../support_beed_patch22
    python examples/compare_tree_support.py --baseline ../support_beed_patch22 --nozzle 0.4 --tree-only

Each checkout/mode runs in a fresh Python process, so module caches cannot mix
old and new code. Models are deterministic millimetre-scale solid boxes; no
downloaded model or mesh Boolean backend is needed. Timings are informative,
not assertions. ``generation_seconds`` measures support generation alone;
``seconds`` also includes the independent geometry checks. Every case uses a
custom detection layer height of 0.25 mm, not the automatic 0.4 mm nozzle profile.
Connectivity uses actual 3D bead centres and radii, including
``z_exact`` for tree beads, rather than their nominal layer heights.
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
import time
import warnings


def make_model(name="bridge"):
    import trimesh

    def block(size, centre):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(centre)
        return mesh

    if name == "bridge":
        parts = [block((12, 8, 1), (0, 0, 6.5))]
        parts += [block((1.5, 8, 6), (x, 0, 3)) for x in (-5.25, 5.25)]
    elif name == "table":
        parts = [block((12, 8, 1), (0, 0, 6.5))]
        parts += [block((1.5, 1.5, 6), (x, y, 3))
                  for x in (-5.25, 5.25) for y in (-3.25, 3.25)]
    else:
        raise ValueError(f"Unknown model: {name}")
    # Each box is watertight and boxes share only their boundary at z=6.
    # The slicer's polygon union resolves the touching components.
    return trimesh.util.concatenate(parts)


def bead_records(plan, layer_height):
    """Extract (x, y, z, diameter), preserving exact tree bead heights."""
    records = []
    if plan is not None:
        for layer in plan.layers:
            default_z = layer.get("z_center")
            if default_z is None:
                default_z = layer["z_bottom"] + 0.5 * layer_height
            records.extend((b["x"], b["y"], b.get("z_exact", default_z), b["d"])
                           for b in layer["beads"])
    return records


def bead_metrics(plan, layer_height, bed_z=0.0):
    """Count duplicates and components whose bead spheres touch the bed.

    Components resting on the model instead of the bed are deliberately not
    called bed-connected. The two fixtures have a clear path to the bed.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    records = bead_records(plan, layer_height)
    n = len(records)
    if not n:
        return dict(beads=0, duplicate_centres=0, components=0,
                    bed_connected_beads=0, floating_beads=0)
    data = np.asarray(records)
    points, radii = data[:, :3], data[:, 3] * 0.5
    duplicate_count = n - len(np.unique(np.round(points, 7), axis=0))
    pairs = cKDTree(points).query_pairs(2 * float(radii.max()) + 1e-6,
                                       output_type="ndarray")
    if len(pairs):
        distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        pairs = pairs[distances <= radii[pairs[:, 0]] + radii[pairs[:, 1]] + 1e-6]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    grounded = np.unique(labels[points[:, 2] - radii <= bed_z + 1e-6])
    bed_connected = int(np.isin(labels, grounded).sum())
    return dict(beads=n, duplicate_centres=int(duplicate_count), components=int(ncomp),
                bed_connected_beads=bed_connected, floating_beads=n - bed_connected)


def mesh_clearance_metrics(plan, layer_height, mesh):
    """Check actual model mesh against bead spheres, independent of slice logic."""
    import numpy as np
    import trimesh

    records = bead_records(plan, layer_height)
    if not records:
        return dict(penetrating_beads=0, beads_below_bed=0, min_bead_bottom_mm=None,
                    centres_below_bed=0, min_bead_centre_z_mm=None)
    data = np.asarray(records)
    centres, radii = data[:, :3], data[:, 3] * 0.5
    query = trimesh.proximity.ProximityQuery(mesh)
    penetrations = 0
    for start in range(0, len(centres), 512):
        batch = centres[start:start + 512]
        _, surface_distance, _ = query.on_surface(batch)
        inside = query.signed_distance(batch) > 1e-6
        overlaps = surface_distance < radii[start:start + 512] - 1e-6
        penetrations += int(np.count_nonzero(inside | overlaps))
    bottoms = centres[:, 2] - radii
    bed_z = float(mesh.bounds[0, 2])
    return dict(penetrating_beads=penetrations,
                beads_below_bed=int(np.count_nonzero(bottoms < bed_z - 1e-6)),
                min_bead_bottom_mm=round(float(bottoms.min()), 7),
                centres_below_bed=int(np.count_nonzero(centres[:, 2] < bed_z - 1e-6)),
                min_bead_centre_z_mm=round(float(centres[:, 2].min()), 7))


def run_worker(repo, mode, model_name, nozzle=2.0, bead_diameter=None):
    # Do not import pellet_support before selecting this checkout.
    sys.path.insert(0, str(repo / "src"))
    from pellet_support.params import SupportGenParams
    from pellet_support.pipeline import generate_support, make_params

    contact, body = make_params(nozzle_diameter_mm=nozzle, bead_diameter_mm=bead_diameter)
    settings = dict(nozzle_diameter_mm=nozzle, layer_height_mm=contact.layer_height_mm(),
                    detection_layer_height_mm=0.25, xy_clearance_mm=0.3,
                    contact_z_gap_mm=0.2, min_island_area_mm2=0.5)
    settings["tree_enabled"] = mode == "tree"
    gen = SupportGenParams(**settings)
    mesh = make_model(model_name)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
        generation_seconds = time.perf_counter() - start
    metrics = bead_metrics(result.plan, gen.layer_height_mm, float(mesh.bounds[0, 2]))
    metrics.update(mesh_clearance_metrics(result.plan, gen.layer_height_mm, mesh))
    metrics.update(mode=mode, tree_enabled=gen.tree_enabled, model=model_name,
                   nozzle_mm=nozzle, bead_diameter_mm=contact.bead_diameter_mm,
                   detection_layer_height_mm=gen.detection_layer_height_mm,
                   xy_clearance_mm=gen.xy_clearance_mm,
                   contact_z_gap_mm=gen.contact_z_gap_mm,
                   generation_seconds=round(generation_seconds, 3),
                   seconds=round(time.perf_counter() - start, 3),
                   tree_stats=getattr(result.plan, "tree_stats", {}),
                   warnings=sorted({str(w.message) for w in caught}))
    print(json.dumps(metrics, ensure_ascii=True))


def compare(repo, baseline, model_names, nozzle=2.0, bead_diameter=None, tree_only=False):
    rows = []
    checkouts = [("current", repo, ("tree",) if tree_only else ("tree", "grid"))]
    if baseline is not None:
        checkouts.insert(0, ("baseline", baseline, ("tree",) if tree_only else ("grid", "tree")))
    for label, checkout, modes in checkouts:
        if not (checkout / "src" / "pellet_support" / "pipeline.py").is_file():
            raise SystemExit(f"Not a support_beed checkout: {checkout}")
        for model_name in model_names:
            for mode in modes:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--repo", str(checkout), "--mode", mode, "--model", model_name,
                           "--nozzle", str(nozzle)]
                if bead_diameter is not None:
                    command += ["--bead-diameter", str(bead_diameter)]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(checkout / "src")
                done = subprocess.run(command, cwd=checkout, env=env, text=True,
                                      capture_output=True, timeout=180)
                if done.returncode:
                    raise SystemExit(f"{label}/{model_name}/{mode} failed:\n{done.stderr}")
                metrics = json.loads(done.stdout)
                metrics["checkout"] = label
                rows.append(metrics)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Path to an unmodified checkout (optional)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=("bridge", "table", "all"), default="all")
    parser.add_argument("--nozzle", type=float, default=2.0, help="Nozzle diameter in mm (default: 2)")
    parser.add_argument("--bead-diameter", type=float, help="Bead diameter in mm (default: half the nozzle)")
    parser.add_argument("--tree-only", action="store_true", help="Skip dense grid generation for fine nozzles")
    parser.add_argument("--json", type=Path, help="Save complete metrics as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("tree", "grid"), default="tree",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.repo.resolve(), args.mode, args.model, args.nozzle, args.bead_diameter)
        return
    models = ("bridge", "table") if args.model == "all" else (args.model,)
    rows = compare(args.repo.resolve(), args.baseline.resolve() if args.baseline else None,
                   models, args.nozzle, args.bead_diameter, args.tree_only)
    print("model    checkout mode       beads duplicates components floating intersects below_bed gen_sec total_sec")
    for row in rows:
        print(f"{row['model']:<8} {row['checkout']:<8} {row['mode']:<9} "
              f"{row['beads']:>6} {row['duplicate_centres']:>10} {row['components']:>10} "
              f"{row['floating_beads']:>8} {row['penetrating_beads']:>10} "
              f"{row['beads_below_bed']:>9} {row['generation_seconds']:>7.3f} "
              f"{row['seconds']:>9.3f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
