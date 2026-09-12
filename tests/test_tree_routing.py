"""Independent geometric checks for routing and retaining support contacts."""

import math
from dataclasses import replace
from pathlib import Path
import sys

import pytest
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pellet_support.params import SupportGenParams
from pellet_support.regions import detect_overhangs
from pellet_support.skeleton import ContactPoint, extract_contact_points, grow_branches, skeleton_to_bead_seeds
from pellet_support.tree_collision import SliceCollision


def _grow(contacts, slices=None, heights=None, bed=0.0, angle=25.0, plate_only=True):
    if heights is None:
        heights = [bed + 0.25 + i * 0.5 for i in range(12)]
    if slices is None:
        slices = [Polygon() for _ in heights]
    gen = SupportGenParams(xy_clearance_mm=0.1, adaptive_bead_size=False,
                           support_on_build_plate_only=plate_only)
    return grow_branches(contacts, slices, heights, gen,
                         step_h=0.4, merge_distance=3.0,
                         max_branch_angle_deg=angle, max_merge_angle_deg=50.0,
                         bed_radius=0.2, bead_radius=0.2)


def _assert_downward_angles(skeleton, angle):
    for parent_idx, child_idx in skeleton.edges():
        parent, child = skeleton.nodes[parent_idx], skeleton.nodes[child_idx]
        drop = child.z - parent.z
        assert drop > 1e-8
        horizontal = math.hypot(parent.x - child.x, parent.y - child.y)
        assert math.degrees(math.atan2(horizontal, drop)) <= angle + 1e-6


def test_plate_only_excludes_model_landing_without_teleporting_other_tree():
    heights = [0.25 + i * 0.5 for i in range(12)]
    slices = [box(-10, -10, 10, 10) if z < 3 else Polygon() for z in heights]
    contacts = [ContactPoint(0, 0, 5, 10, 1), ContactPoint(20, 0, 5, 10, 1)]
    skeleton = _grow(contacts, slices, heights)
    seeds = skeleton_to_bead_seeds(skeleton, 0.4, 0.4, include_on_model=False)
    assert seeds
    assert all(seed[0] == pytest.approx(20) for seed in seeds)
    assert sum(node.on_bed for node in skeleton.nodes) == 1
    _assert_downward_angles(skeleton, 25)
    collision = SliceCollision(slices, heights, 0.1)
    for parent_idx, child_idx in skeleton.edges():
        parent, child = skeleton.nodes[parent_idx], skeleton.nodes[child_idx]
        assert collision.edge_clear((child.x, child.y, child.z),
                                    (parent.x, parent.y, parent.z),
                                    child.radius, parent.radius)


def test_zero_branch_angle_preserves_separate_vertical_columns():
    contacts = [ContactPoint(-0.8, 0, 4, 8, 1), ContactPoint(0.8, 0, 4, 8, 1)]
    skeleton = _grow(contacts, angle=0)
    assert sum(node.kind == "contact" for node in skeleton.nodes) == 2
    assert len(skeleton.roots()) == 2
    assert all(node.x == pytest.approx(-0.8) or node.x == pytest.approx(0.8)
               for node in skeleton.nodes)
    _assert_downward_angles(skeleton, 0)


def test_contacts_at_different_heights_merge_without_horizontal_edges():
    contacts = [ContactPoint(-1, 0, 5, 10, 1), ContactPoint(1, 0, 3.65, 7, 1)]
    skeleton = _grow(contacts)
    assert sum(node.kind == "contact" for node in skeleton.nodes) == 2
    assert len(skeleton.roots()) == 1
    assert any(len(children) == 2 for children in skeleton.children)
    _assert_downward_angles(skeleton, 25)


def test_translated_bed_preserves_patch22_root_compression():
    bed = 10.0
    skeleton = _grow([ContactPoint(1, 2, 14, 8, 1)], bed=bed)
    assert skeleton.nodes
    for root_idx in skeleton.roots():
        root = skeleton.nodes[root_idx]
        assert root.on_bed
        assert root.z == pytest.approx(bed + 0.8 * 0.2, abs=1e-8)
    assert all(node.z >= bed for node in skeleton.nodes)


def test_model_landing_is_retained_only_when_build_plate_only_is_disabled():
    heights = [0.5 + i for i in range(6)]
    slices = [box(-10, -10, 10, 10) if z < 3 else Polygon() for z in heights]
    contacts = [ContactPoint(0, 0, 4.8, 4, 1)]
    landed = _grow(contacts, slices, heights, angle=0, plate_only=False)
    assert sum(node.kind == "contact" for node in landed.nodes) == 1
    assert len(landed.roots()) == 1
    root = landed.nodes[landed.roots()[0]]
    assert root.z - root.radius == pytest.approx(3.0, abs=1e-6)
    _assert_downward_angles(landed, 0)
    assert root.on_model
    assert skeleton_to_bead_seeds(landed, 0.4, 0.4, include_on_model=False) == []


def test_continuous_overhang_layers_share_contacts_but_separate_shelves_do_not():
    region = box(0, 0, 4, 2)
    sampling = dict(max_area_per_point=4.0, min_area=0.1, contact_spacing_mm=2.0)
    baseline = extract_contact_points([region], [0], **sampling)
    continuous = extract_contact_points([region] * 5, [0, 0.1, 0.2, 0.3, 0.4], **sampling)
    separated = extract_contact_points([region, region, Polygon(), region, region],
                                       [0, 0.1, 0.2, 0.3, 0.4], **sampling)
    assert baseline
    assert len(continuous) == len(baseline)
    assert len(separated) == 2 * len(baseline)
    assert {point.layer for point in separated} == {0, 3}
    assert all(region.covers(Point(point.x, point.y)) for point in separated)


def test_detect_overhangs_removes_closed_cavity_but_keeps_external_overhang():
    solid = box(0, 0, 10, 10)
    ring = solid.difference(box(2, 2, 8, 8))
    external_ledge = box(10, 2, 14, 8)
    tiny_island = box(20, 20, 20.5, 20.5)
    roof = solid.union(external_ledge).union(tiny_island)
    slices = [solid, ring, ring, roof]
    gen = SupportGenParams(overhang_angle_deg=45, min_island_area_mm2=2.0,
                           allow_internal_supports=False, removal_opening_mm=0.5)
    filtered = detect_overhangs(slices, gen, 0.5)
    allowed = detect_overhangs(slices, replace(gen, allow_internal_supports=True), 0.5)
    assert allowed[-1].covers(Point(5, 5))
    assert not filtered[-1].covers(Point(5, 5))
    assert filtered[-1].covers(Point(12, 5))
    assert not filtered[-1].intersects(tiny_island)
    assert filtered[-1].is_valid
    parts = filtered[-1].geoms if hasattr(filtered[-1], "geoms") else [filtered[-1]]
    assert all(part.area >= gen.min_island_area_mm2 for part in parts)
