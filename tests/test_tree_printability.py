"""A bead must be supported when printed, not merely connected from above."""

import pytest
import trimesh

from pellet_support.skeleton import prune_floating


def _box(extents, centre):
    model = trimesh.creation.box(extents=extents)
    model.apply_translation(centre)
    return model


def _column(x=0.0, y=0.0, bottom=0.08, diameter=0.2, count=5):
    return [(x, y, bottom + i * diameter * 0.8, diameter) for i in range(count)]


def _bed_arch():
    """Two bed columns meet through rising, overlapping bead chains."""
    beads = []
    for side in (0.0, 0.96):
        beads.extend(_column(x=side, bottom=0.16, diameter=0.4, count=6))
        direction = 1 if side == 0.0 else -1
        beads.extend((side + direction * 0.12 * i, 0.0, 1.76 + 0.32 * i, 0.4)
                     for i in range(1, 5))
    # The two rising paths share their final bead at the arch crown.
    return list(dict.fromkeys(beads))


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_bed_connected_arch_does_not_anchor_a_downward_hanging_tail(reverse):
    arch = _bed_arch()
    tail = [(0.48, 0.0, 3.04 - 0.32 * i, 0.4) for i in range(1, 9)]
    beads = arch + tail
    if reverse:
        beads.reverse()
    kept, dropped = prune_floating(beads, None, bed_z=0.0, xy_clearance=0.3)
    # These beads connect to the bed only by first climbing the hanging tail.
    # No model or previously printed bead supports the bottom of that tail.
    unsupported_tail = [bead for bead in tail if bead[2] < 1.44]
    assert unsupported_tail
    assert all(bead not in kept for bead in unsupported_tail)
    assert all(bead in kept for bead in arch)
    assert dropped == len(beads) - len(kept)


@pytest.mark.parametrize("model", [
    _box((0.2, 2.0, 2.0), (-0.45, 0.0, 1.0)),
    _box((0.2, 2.0, 0.2), (-0.45, 0.0, 0.75)),
], ids=["vertical-wall", "diagonally-lower-corner"])
def test_nearby_model_wall_or_corner_is_not_a_surface_below_the_bead(model):
    beads = _column(bottom=1.0)
    kept, dropped = prune_floating(beads, model, bed_z=0.0, xy_clearance=0.3)
    assert kept == []
    assert dropped == len(beads)


@pytest.mark.parametrize("xy_clearance", [0.0, 0.8])
def test_small_beads_one_mm_above_a_model_are_not_grounded(xy_clearance):
    model = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
    # The slab ends at z=1.0; the first 0.2 mm bead's bottom is z=2.0.
    beads = _column(bottom=2.1)
    kept, dropped = prune_floating(beads, model, bed_z=0.0, xy_clearance=xy_clearance)
    assert kept == []
    assert dropped == len(beads)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_column_actually_resting_on_a_flat_model_is_preserved(reverse):
    model = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
    beads = _column(bottom=1.1)
    if reverse:
        beads.reverse()
    kept, dropped = prune_floating(beads, model, bed_z=0.0, xy_clearance=0.3)
    assert kept == beads
    assert dropped == 0


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_tree_with_upward_support_paths_from_the_bed_is_preserved(reverse):
    beads = _bed_arch()
    if reverse:
        beads.reverse()
    kept, dropped = prune_floating(beads, None, bed_z=0.0, xy_clearance=0.3)
    assert kept == beads
    assert dropped == 0


def test_empty_seed_input_is_preserved():
    assert prune_floating([], None, bed_z=0.0, xy_clearance=0.3) == ([], 0)
