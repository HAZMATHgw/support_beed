"""Repair missing lower support while retaining the intended upper beads."""

import numpy as np
import pytest
import trimesh

from pellet_support.printability import repair_support_paths
from pellet_support.skeleton import prune_floating


DIAMETER = 0.2


def _box(extents, centre):
    model = trimesh.creation.box(extents=extents)
    model.apply_translation(centre)
    return model


def _column(x=0.0, bottom=0.08, count=5):
    return [(x, 0.0, bottom + i * 0.16, DIAMETER) for i in range(count)]


def _arch_with_tail():
    arch = []
    for side in (0.0, 0.48):
        arch.extend(_column(x=side, count=6))
        direction = 1 if side == 0.0 else -1
        arch.extend((side + direction * 0.06 * i, 0.0, 0.88 + 0.16 * i, DIAMETER)
                    for i in range(1, 5))
    arch = list(dict.fromkeys(arch))
    tail = [(0.24, 0.0, 1.52 - 0.16 * i, DIAMETER) for i in range(1, 9)]
    return arch, tail


def _repair(beads, model=None, **kwargs):
    return repair_support_paths(beads, model, bed_z=0.0,
                                bead_diameter=DIAMETER, xy_clearance=0.05,
                                z_gap=0.3, **kwargs)


def _assert_printable_without_pruning(beads, model=None):
    kept, dropped = prune_floating(beads, model, bed_z=0.0, xy_clearance=0.05)
    assert kept == beads
    assert dropped == 0


def _assert_no_sphere_intersections(beads, model):
    data = np.asarray(beads)
    _, distance, _ = model.nearest.on_surface(data[:, :3])
    assert np.all(distance >= 0.5 * data[:, 3] - 1e-7)
    assert np.all(model.nearest.signed_distance(data[:, :3]) <= 1e-7)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_adds_a_real_model_foot_without_moving_or_dropping_original_beads(reverse):
    model = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
    # Slab top is z=1; first sphere bottom is z=1.3, leaving a 0.3 mm air gap.
    beads = _column(bottom=1.4)
    if reverse:
        beads.reverse()
    assert prune_floating(beads, model, 0.0, 0.05)[0] == []
    repaired, added_count = _repair(beads, model)
    assert repaired[:len(beads)] == beads
    assert added_count == len(repaired) - len(beads) > 0
    added = repaired[len(beads):]
    assert any(abs(z - d * 0.5 - 1.0) < 1e-7 for x, y, z, d in added)
    _assert_printable_without_pruning(repaired, model)
    _assert_no_sphere_intersections(repaired, model)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_hanging_tail_is_saved_by_a_new_lower_printable_path(reverse):
    arch, tail = _arch_with_tail()
    beads = arch + tail
    if reverse:
        beads.reverse()
    before, _ = prune_floating(beads, None, 0.0, 0.05)
    assert any(bead not in before for bead in tail)
    repaired, added_count = _repair(beads)
    assert repaired[:len(beads)] == beads
    assert added_count == len(repaired) - len(beads) > 0
    assert min(bead[2] for bead in repaired[len(beads):]) < min(bead[2] for bead in tail)
    _assert_printable_without_pruning(repaired)


@pytest.mark.parametrize("extra_budget", [0, 1, 2])
def test_repair_respects_the_remaining_bead_budget(extra_budget):
    model = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
    beads = _column(bottom=1.4)
    limit = len(beads) + extra_budget
    repaired, added_count = _repair(beads, model, max_beads=limit)
    assert repaired[:len(beads)] == beads
    assert len(repaired) <= limit
    assert added_count == len(repaired) - len(beads)
    if extra_budget == 0:
        assert repaired == beads
    if extra_budget == 2:
        assert added_count > 0
        _assert_printable_without_pruning(repaired, model)


def test_reconnection_does_not_cross_a_wall_to_reach_a_lower_branch():
    wall = _box((0.1, 2.0, 2.0), (0.0, 0.0, 1.0))
    supported_left = _column(x=-0.35, count=8)
    unsupported_right = _column(x=0.35, bottom=1.2)
    beads = supported_left + unsupported_right
    # Lower left beads are within the search and slope limits, but a wall
    # intersects those shortcuts. A vertical path on the right remains clear.
    repaired, added_count = _repair(beads, wall, allow_model=False)
    assert repaired[:len(beads)] == beads
    assert added_count > 0
    assert all(bead[0] >= 0.35 - 1e-7 for bead in repaired[len(beads):])
    _assert_printable_without_pruning(repaired)
    _assert_no_sphere_intersections(repaired, wall)


@pytest.mark.parametrize("allow_model", [False, True])
def test_build_plate_only_does_not_accept_an_existing_model_foundation(allow_model):
    model = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
    beads = _column(bottom=1.1)
    repaired, added_count = _repair(beads, model, allow_model=allow_model)
    assert repaired == beads
    assert added_count == 0
    kept, _ = prune_floating(repaired, model if allow_model else None, 0.0, 0.05)
    assert kept == (beads if allow_model else [])


@pytest.mark.parametrize("beads", [[], _column()], ids=["empty", "already-grounded"])
def test_repair_does_not_change_already_valid_input(beads):
    repaired, added_count = _repair(beads)
    assert repaired == beads
    assert added_count == 0
