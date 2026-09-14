"""Geometric invariants of branch smoothing and shallow contact placement."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pellet_support.params import SupportGenParams
from pellet_support.skeleton import (
    ContactPoint, SupportSkeleton, grow_branches, smooth_branches,
)
from pellet_support.tree_collision import SliceCollision


def _chain(points):
    skeleton = SupportSkeleton()
    parent = None
    for i, (x, y, z) in enumerate(points):
        kind = "contact" if i == len(points) - 1 else "trunk"
        parent = skeleton.add_node(x, y, z, parent, 0.025, i, kind=kind)
    skeleton.nodes[0].on_bed = True
    return skeleton


def _coordinates(skeleton):
    return np.array([(n.x, n.y, n.z) for n in skeleton.nodes])


def _length(skeleton):
    points = _coordinates(skeleton)
    return sum(np.linalg.norm(points[c] - points[p]) for p, c in skeleton.edges())


def _assert_angles(skeleton, limit):
    for p, c in skeleton.edges():
        parent, child = skeleton.nodes[p], skeleton.nodes[c]
        dz = child.z - parent.z
        assert dz > 0
        assert math.degrees(math.atan2(math.hypot(child.x - parent.x,
                                                child.y - parent.y), dz)) <= limit + 1e-7


def _smooth(skeleton, **kwargs):
    heights = [0.05 + i * 0.1 for i in range(30)]
    return smooth_branches(skeleton, [Polygon() for _ in heights], heights,
                           xy_clearance=0.0, **kwargs)


def test_uneven_heights_do_not_bend_an_already_straight_branch():
    # The old XY midpoint moved the second node from x=.04 to x=.12,
    # increasing its lower edge from 21.8 degrees to 50.2 degrees.
    skeleton = _chain([(0, 0, 0.1), (0.04, 0, 0.2), (0.4, 0, 1.1)])
    before = _coordinates(skeleton).copy()
    _smooth(skeleton)
    np.testing.assert_allclose(_coordinates(skeleton), before, atol=1e-12)
    _assert_angles(skeleton, 25)


@pytest.mark.parametrize("limit", [10.0, 25.0, 40.0])
def test_smoothing_shortens_uneven_zigzag_and_preserves_edge_angles(limit):
    lean = math.tan(math.radians(limit))
    skeleton = _chain([(0, 0, 0.1), (0.08 * lean, 0, 0.2),
                       (-0.10 * lean, 0, 0.7), (0.12 * lean, 0, 1.2),
                       (0, 0, 2.1)])
    before = _length(skeleton)
    endpoints = _coordinates(skeleton)[[0, -1]].copy()
    edges = skeleton.edges()
    _assert_angles(skeleton, limit)
    _smooth(skeleton, iterations=8, max_branch_angle_deg=limit)
    assert _length(skeleton) < before
    _assert_angles(skeleton, limit)
    np.testing.assert_array_equal(_coordinates(skeleton)[[0, -1]], endpoints)
    assert skeleton.edges() == edges


def test_model_landing_and_merge_junction_remain_fixed():
    skeleton = _chain([(0, 0, 0.1), (0.15, 0, 1.1), (0, 0, 2.1)])
    skeleton.nodes[1].on_model = True
    before = _coordinates(skeleton).copy()
    _smooth(skeleton)
    np.testing.assert_array_equal(_coordinates(skeleton), before)
    skeleton.nodes[1].on_model = False
    skeleton.add_node(0.2, 0, 2.1, 1, 0.025, 2, kind="contact")
    before = _coordinates(skeleton).copy()
    _smooth(skeleton)
    np.testing.assert_array_equal(_coordinates(skeleton), before)


def test_smoothing_keeps_clear_detour_when_shortcut_hits_model():
    skeleton = _chain([(0, 0, 0.1), (0.35, 0, 1.1), (0, 0, 2.1)])
    heights = [0.05 + i * 0.1 for i in range(24)]
    slices = [box(0.14, -0.04, 0.23, 0.04) if 1.0 <= z <= 1.2 else Polygon()
              for z in heights]
    collision = SliceCollision(slices, heights, 0.0)
    before = _coordinates(skeleton).copy()
    for p, c in skeleton.edges():
        assert collision.edge_clear(before[p], before[c], 0.025, 0.025)
    smooth_branches(skeleton, slices, heights, xy_clearance=0.0)
    np.testing.assert_array_equal(_coordinates(skeleton), before)


@pytest.mark.parametrize("kwargs", [
    {"relax": float("nan")}, {"relax": -0.1}, {"relax": 1.1},
    {"max_branch_angle_deg": float("inf")}, {"max_branch_angle_deg": -1},
    {"max_branch_angle_deg": 90},
])
def test_invalid_smoothing_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _smooth(SupportSkeleton(), **kwargs)


def _shallow(contacts, *, offset=2.0, blocked=False, bed=0.0):
    heights = [bed + 0.05 + i * 0.1 for i in range(12)]
    slices = [box(-1, -1, 1, 1) if blocked else Polygon() for _ in heights]
    return grow_branches(
        contacts, slices, heights,
        SupportGenParams(xy_clearance_mm=0.0),
        step_h=0.1, merge_distance=0.4, bed_radius=0.2, bead_radius=0.2,
        contact_z_offset=offset,
    )


@pytest.mark.parametrize("bed", [0.0, 10.0])
def test_shallow_contact_survives_offset_larger_than_available_height(bed):
    skeleton = _shallow([ContactPoint(0, 0, bed + 0.5, 5, 1)], bed=bed)
    assert len(skeleton.nodes) == 1
    node = skeleton.nodes[0]
    assert node.kind == "contact" and node.on_bed
    assert node.z == pytest.approx(bed + 0.16)
    assert node.contact_height == pytest.approx(bed + 0.5)
    assert skeleton.skipped_contacts == 0


def test_contact_exactly_at_bed_center_is_added_only_once():
    skeleton = _shallow([ContactPoint(0, 0, 0.46, 4, 1)], offset=0.30)
    assert len(skeleton.nodes) == 1
    assert skeleton.nodes[0].on_bed
    assert skeleton.nodes[0].contact_height == 0.46


@pytest.mark.parametrize("height, blocked", [(0.1, False), (0.5, True)])
def test_near_bed_contact_without_free_bead_space_is_counted_as_skipped(height, blocked):
    cp = ContactPoint(0, 0, height, 1, 1)
    skeleton = _shallow([cp], blocked=blocked)
    assert skeleton.nodes == []
    assert skeleton.skipped_contacts == 1
    assert skeleton.dropped_points == [(0, 0, height)]


def test_regular_contact_keeps_original_ceiling_height():
    skeleton = _shallow([ContactPoint(0, 0, 1.0, 9, 1)], offset=0.3)
    contacts = [n for n in skeleton.nodes if n.kind == "contact"]
    assert len(contacts) == 1
    assert contacts[0].z == pytest.approx(0.7)
    assert contacts[0].contact_height == 1.0
