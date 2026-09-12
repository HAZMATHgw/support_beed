"""Geometric regressions for sparse, connected bead tree supports."""

import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

# Also support `python -m pytest` before installing this src-layout package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.compare_tree_support import bead_metrics, make_model, mesh_clearance_metrics
from pellet_support.meshing import BeadPlan
from pellet_support.params import SupportGenParams
from pellet_support.pipeline import generate_support, make_params
from pellet_support.skeleton import (
    ContactPoint,
    SupportSkeleton,
    extract_contact_points,
    grow_branches,
    skeleton_to_bead_seeds,
)


@pytest.mark.parametrize("region", [
    Polygon([(0, 0), (8, 0), (8, 8), (0, 8)],
            holes=[[(1, 1), (1, 7), (7, 7), (7, 1)]]),
    Polygon([(0, 0), (6, 0), (6, 1), (1, 1), (1, 6), (0, 6)]),
], ids=["hole", "concave"])
def test_single_contact_is_inside_its_overhang(region):
    contacts = extract_contact_points([region], [5.0], max_area_per_point=1000)
    assert len(contacts) == 1
    assert region.contains(Point(contacts[0].x, contacts[0].y))


@pytest.mark.parametrize("spacing", [None, 4.0], ids=["legacy-area", "sparse-spacing"])
def test_every_disconnected_island_keeps_a_contact_when_grid_misses_thin_shape(spacing):
    square = box(0, 0, 2, 2)
    thin_l = Polygon([(10, 0), (13, 0), (13, 0.1),
                      (10.1, 0.1), (10.1, 3), (10, 3)])
    contacts = extract_contact_points([MultiPolygon([square, thin_l])], [5.0],
                                      max_area_per_point=0.5, min_area=0.01,
                                      contact_spacing_mm=spacing)
    for island in (square, thin_l):
        assert any(island.contains(Point(cp.x, cp.y)) for cp in contacts)
    assert all(square.contains(Point(cp.x, cp.y)) or thin_l.contains(Point(cp.x, cp.y))
               for cp in contacts)


def test_larger_spacing_reduces_contacts_and_sampling_is_repeatable():
    region = box(0, 0, 16, 12)
    dense = extract_contact_points([region], [6.0], max_area_per_point=3.0,
                                   contact_spacing_mm=2.0)
    sparse = extract_contact_points([region], [6.0], max_area_per_point=3.0,
                                    contact_spacing_mm=4.0)
    again = extract_contact_points([region], [6.0], max_area_per_point=3.0,
                                   contact_spacing_mm=4.0)
    assert 0 < len(sparse) < len(dense)
    assert sparse == again
    assert all(region.contains(Point(cp.x, cp.y)) for cp in sparse)


def test_nearby_tips_merge_only_downwards_within_branch_angle():
    contacts = [ContactPoint(-1, 0, 4, 8, 1), ContactPoint(1, 0, 4, 8, 1)]
    heights = np.arange(0, 4.25, 0.5)
    angle = 25.0
    skeleton = grow_branches(contacts, [Polygon() for _ in heights], heights,
                             SupportGenParams(xy_clearance_mm=0.0),
                             step_h=0.5, merge_distance=6.0,
                             max_branch_angle_deg=angle)
    assert sum(n.kind == "contact" for n in skeleton.nodes) == 2
    assert len(skeleton.roots()) == 1
    for parent_idx, child_idx in skeleton.edges():
        parent, child = skeleton.nodes[parent_idx], skeleton.nodes[child_idx]
        drop = child.z - parent.z
        assert drop > 1e-8, "A merge must not create a horizontal or upward edge"
        lean = math.hypot(child.x - parent.x, child.y - parent.y)
        assert math.degrees(math.atan2(lean, drop)) <= angle + 1e-6


def test_shared_junction_beads_are_unique_and_short_edges_form_connected_chains():
    skeleton = SupportSkeleton()
    root = skeleton.add_node(0, 0, 0.5, None, 0.5, 0, kind="root")
    junction = skeleton.add_node(0, 0, 1.7, root, 0.5, 1)
    skeleton.add_node(-0.2, 0, 2.9, junction, 0.5, 2, kind="contact")
    skeleton.add_node(0.2, 0, 2.9, junction, 0.5, 2, kind="contact")
    seeds = skeleton_to_bead_seeds(skeleton, 1.0, 1.0)
    points = np.asarray(seeds)[:, :3]
    assert len(points) == len(np.unique(np.round(points, 7), axis=0))
    plan = BeadPlan(layers=[dict(z_bottom=0, beads=[
        dict(x=x, y=y, z_exact=z, d=d) for x, y, z, d in seeds
    ])])
    metrics = bead_metrics(plan, 1.0)
    assert metrics["components"] == 1
    assert metrics["floating_beads"] == 0


@pytest.mark.parametrize("name", ["bridge", "table"])
def test_tree_pipeline_is_sparse_unique_and_bed_connected(name):
    contact, body = make_params(nozzle_diameter_mm=2.0, bead_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=2.0, tree_enabled=True,
                           layer_height_mm=contact.layer_height_mm(),
                           detection_layer_height_mm=0.25, xy_clearance_mm=0.3,
                           contact_z_gap_mm=0.2, min_island_area_mm2=0.5)
    model = make_model(name)
    tree = generate_support(model, gen, contact, body, detail=0, verbose=False)
    grid = generate_support(model, replace(gen, tree_enabled=False), contact, body,
                            detail=0, verbose=False)
    tree_metrics = bead_metrics(tree.plan, gen.layer_height_mm)
    grid_metrics = bead_metrics(grid.plan, gen.layer_height_mm)
    assert 0 < tree_metrics["beads"] < grid_metrics["beads"] * 0.5
    assert tree_metrics["duplicate_centres"] == 0
    assert tree_metrics["floating_beads"] == 0
    clearance = mesh_clearance_metrics(tree.plan, gen.layer_height_mm, model)
    assert clearance["penetrating_beads"] == 0
    # Patch-22 intentionally presses root spheres slightly into the bed.
    assert clearance["centres_below_bed"] == 0
    assert len(tree.mesh.faces) > 0
    assert tree.mesh.is_watertight


@pytest.mark.parametrize("name", ["bridge", "table"])
def test_point_four_mm_nozzle_keeps_connected_collision_free_support(name):
    contact, body = make_params(nozzle_diameter_mm=0.4)
    gen = SupportGenParams(nozzle_diameter_mm=0.4, tree_enabled=True,
                           layer_height_mm=contact.layer_height_mm(),
                           detection_layer_height_mm=0.25, xy_clearance_mm=0.3,
                           contact_z_gap_mm=0.2, min_island_area_mm2=0.5)
    model = make_model(name)
    tree = generate_support(model, gen, contact, body, detail=0, verbose=False)
    metrics = bead_metrics(tree.plan, gen.layer_height_mm)
    assert metrics["beads"] > 0
    assert metrics["duplicate_centres"] == 0
    assert metrics["floating_beads"] == 0
    clearance = mesh_clearance_metrics(tree.plan, gen.layer_height_mm, model)
    assert clearance["penetrating_beads"] == 0
    assert clearance["centres_below_bed"] == 0
