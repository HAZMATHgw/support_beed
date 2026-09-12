import math

import pytest
from shapely.geometry import Point, Polygon, box

from pellet_support.tree_collision import SliceCollision


def test_empty_stack_and_empty_slices_are_clear():
    for collision in (SliceCollision([], [], 0.2),
                      SliceCollision([Polygon(), None], [0.5, 1.5], 0.2)):
        assert collision.sphere_clear(0, 0, 0, 10)
        assert collision.edge_clear((0, 0, 0), (10, 0, 0), 1, 2)
        assert collision.max_radius(0, 0, 0, 10) == 10


@pytest.mark.parametrize("slices, heights, clearance", [
    ([box(0, 0, 1, 1)], [], 0),
    ([None, None], [1, 1], 0),
    ([None, None], [1, 0], 0),
    ([None], [math.nan], 0),
    ([], [], -1),
])
def test_invalid_stack_is_rejected(slices, heights, clearance):
    with pytest.raises(ValueError):
        SliceCollision(slices, heights, clearance)


def test_sphere_uses_slab_extent_and_actual_cross_section():
    # The filled slice occupies z=1..2. At z=0.2, a unit sphere has a
    # maximum XY radius of 0.6 in that slab, rather than its full radius 1.
    collision = SliceCollision([None, box(0, 0, 10, 10)], [0.5, 1.5], 0.1)
    assert collision.sphere_clear(-0.71, 5, 0.2, 1)
    assert not collision.sphere_clear(-0.69, 5, 0.2, 1)
    assert not collision.sphere_clear(5, 5, 0.99, 0.02)


def test_contact_below_model_is_not_pushed_out_sideways():
    collision = SliceCollision([None, box(-5, -5, 5, 5)], [0.5, 1.5], 0.8)
    assert collision.sphere_clear(0, 0, 0.4, 0.5)
    assert collision.edge_clear((0, 0, 0.4), (1, 0, -1), 0.5, 0.5)
    assert collision.sphere_clear(0, 0, 0.5, 0.5)  # tangent, no penetration
    assert not collision.sphere_clear(0, 0, 0.51, 0.5)


def test_intermediate_wall_cannot_be_skipped_by_clear_endpoints():
    wall = box(-0.00001, -10, 0.00001, 10)
    collision = SliceCollision([wall, wall], [0.5, 1.5], 0)
    start, end = (-3.2, 0, 1), (7.7, 0, 1)
    assert collision.sphere_clear(*start, 0.1)
    assert collision.sphere_clear(*end, 0.1)
    assert not collision.edge_clear(start, end, 0.1, 0.1)


def test_vertical_edge_and_hole_respect_clearance():
    ring = Polygon(box(-5, -5, 5, 5).exterior.coords,
                   [box(-2, -2, 2, 2).exterior.coords])
    collision = SliceCollision([ring, ring], [0.5, 1.5], 0.1)
    assert collision.edge_clear((0, 0, -2), (0, 0, 4), 1, 1)
    assert not collision.edge_clear((1.1, 0, -2), (1.1, 0, 4), 1, 1)


def test_varying_radius_cannot_skip_a_middle_slab():
    collision = SliceCollision([None, box(0, -5, 5, 5), None], [0.5, 1.5, 2.5], 0.1)
    assert not collision.edge_clear((-0.6, 0, -1), (-0.6, 0, 4), 0.1, 2)
    assert collision.edge_clear((-3, 0, -1), (-3, 0, 4), 0.1, 2)


def test_max_radius_at_wall_and_under_roof():
    wall = SliceCollision([box(0, -5, 5, 5)] * 2, [0.5, 1.5], 0.2)
    assert wall.max_radius(-2, 0, 1, 5) == pytest.approx(1.8, abs=1e-8)
    assert wall.max_radius(2, 0, 1, 5) == 0
    roof = SliceCollision([None, box(-5, -5, 5, 5)], [0.5, 1.5], 0.8)
    assert roof.max_radius(0, 0, 0, 5) == pytest.approx(1.0, abs=1e-8)


def test_free_region_keeps_bridge_contacts_away_from_columns():
    columns = box(-4, -2, -2, 2).union(box(2, -2, 4, 2))
    collision = SliceCollision([columns, columns], [0.5, 1.5], 0.1)
    free = collision.free_region(box(-2, -1, 2, 1), 1.2, 0.2)
    assert free.is_valid and not free.is_empty
    assert free.contains(Point(0, 0))
    assert not free.covers(Point(-1.8, 0))
    assert not free.covers(Point(1.8, 0))
    assert free.area > 6.7  # retain nearly all of the physically usable span
    for x, y in free.exterior.coords:
        assert collision.sphere_clear(x, y, 1.2, 0.2)


def test_free_region_uses_cross_sections_and_skips_tangent_roof():
    collision = SliceCollision([None, box(0, 0, 10, 10)], [0.5, 1.5], 0.1)
    free = collision.free_region(box(-2, 4, 1, 6), 0.2, 1)
    assert free.covers(Point(-0.71, 5))
    assert not free.covers(Point(-0.69, 5))
    roof_contact = box(2, 2, 8, 8)
    assert collision.free_region(roof_contact, 0.5, 0.5).equals(roof_contact)


def test_free_region_handles_empty_invalid_and_fully_blocked_regions():
    collision = SliceCollision([box(0, 0, 10, 10)], [1], 0.1)
    assert collision.free_region(None, 1, 0.5).is_empty
    assert collision.free_region(Polygon(), 1, 0.5).is_empty
    assert collision.free_region(box(2, 2, 8, 8), 1, 0.5).is_empty
    bowtie = Polygon([(-4, -4), (-2, -2), (-4, -2), (-2, -4), (-4, -4)])
    assert not bowtie.is_valid
    assert collision.free_region(bowtie, 1, 0.5).is_valid


def test_single_slice_and_nonuniform_heights_have_documented_bounds():
    single = SliceCollision([box(0, 0, 1, 1)], [4], 0)
    assert single.sphere_clear(0.5, 0.5, 3, 1)
    assert not single.sphere_clear(0.5, 0.5, 3, 1.01)
    collision = SliceCollision([None] * 3, [1, 3, 6], 0)
    assert collision.slab_bottoms == (0, 2, 4.5)
    assert collision.slab_tops == (2, 4.5, 7.5)


@pytest.mark.parametrize("method, arguments", [
    ("sphere_clear", (0, 0, math.inf, 1)),
    ("sphere_clear", (0, 0, 0, -1)),
    ("edge_clear", ((0, 0), (0, 0, 1), 1, 1)),
    ("edge_clear", ((0, 0, 0), (0, 0, 1), 1, math.nan)),
    ("max_radius", (0, 0, 0, -1)),
])
def test_invalid_queries_are_rejected(method, arguments):
    with pytest.raises(ValueError):
        getattr(SliceCollision([], [], 0), method)(*arguments)
