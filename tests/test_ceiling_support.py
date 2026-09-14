"""Closing an upper contact must preserve its printable path from below."""

import numpy as np
import pytest
import trimesh

from pellet_support.printability import close_ceiling_gaps, supported_mask


DIAMETER = 0.2


def _box(extents, centre):
    model = trimesh.creation.box(extents=extents)
    model.apply_translation(centre)
    return model


def _ceiling(z, thickness=0.16):
    return _box((2.0, 2.0, thickness), (0.0, 0.0, z + thickness / 2))


def _column(bottom=0.08, diameter=DIAMETER, count=5):
    return [(0.0, 0.0, bottom + i * diameter * 0.8, diameter)
            for i in range(count)]


def _close(beads, model, targets, **kwargs):
    return close_ceiling_gaps(beads, model, targets,
                              bead_diameter=DIAMETER,
                              xy_clearance=0.02, **kwargs)[:2]


def _assert_preserved_and_supported(original, closed, model):
    assert closed[:len(original)] == original
    assert supported_mask(closed, model, bed_z=0.0).all()


def _assert_flat_contact(beads, ceiling_z, embed=0.0):
    data = np.asarray(beads)
    assert np.isclose(data[:, 2] + data[:, 3] * 0.5,
                      ceiling_z + embed, atol=1e-7, rtol=0).any()


def _assert_no_intersections(beads, obstacle):
    data = np.asarray(beads)
    _, distance, _ = obstacle.nearest.on_surface(data[:, :3])
    assert np.all(distance >= data[:, 3] * 0.5 - 1e-7)
    assert np.all(obstacle.nearest.signed_distance(data[:, :3]) <= 1e-7)


@pytest.mark.parametrize("embed", [0.0, 0.015])
def test_short_ceiling_gap_does_not_sever_the_last_beads_lower_support(embed):
    beads = _column()
    model = _ceiling(0.94)
    assert supported_mask(beads, model).all()

    # Lifting the z=.72 tip to z=.84 leaves it .28 mm above its lower
    # neighbour: the two .2 mm spheres no longer touch.
    closed, changed = _close(beads, model, [(0.0, 0.0, 0.94)], embed=embed)

    assert changed > 0
    _assert_preserved_and_supported(beads, closed, model)
    _assert_flat_contact(closed, 0.94, embed)


@pytest.mark.parametrize("reverse_targets", [False, True])
def test_contacts_on_two_shelves_sharing_xy_both_reach_their_own_ceiling(reverse_targets):
    lower_ceiling = _ceiling(0.94)
    model = trimesh.util.concatenate([lower_ceiling, _ceiling(1.9)])
    # The upper column rests on the lower slab's top at z=1.10.
    beads = _column() + _column(bottom=1.20, count=3)
    targets = [(0.0, 0.0, 0.94), (0.0, 0.0, 1.9)]
    if reverse_targets:
        targets.reverse()
    assert supported_mask(beads, model).all()

    closed, changed = _close(beads, model, targets)

    assert changed > 0
    _assert_preserved_and_supported(beads, closed, model)
    _assert_flat_contact(closed, 0.94)
    _assert_flat_contact(closed, 1.9)
    _assert_no_intersections(closed, model)


def test_larger_body_beads_connect_to_a_small_tip_and_touch_with_their_actual_radius():
    beads = _column(bottom=0.016, diameter=0.04, count=20)
    model = _ceiling(0.85)
    assert supported_mask(beads, model).all()

    closed, changed = _close(beads, model, [(0.0, 0.0, 0.85)])

    assert changed > 0
    _assert_preserved_and_supported(beads, closed, model)
    _assert_flat_contact(closed, 0.85)
    _assert_no_intersections(closed, model)


def test_a_tip_on_a_sloping_ceiling_touches_without_cutting_into_the_model():
    angle = np.deg2rad(30.0)
    model = trimesh.creation.box(extents=(2.0, 2.0, 0.16))
    model.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    model.apply_translation([0.0, 0.0, 0.94 + 0.08 / np.cos(angle)])
    beads = _column()

    closed, changed = _close(beads, model, [(0.0, 0.0, 0.94)])

    assert changed > 0
    _assert_preserved_and_supported(beads, closed, model)
    _assert_no_intersections(closed, model)
    data = np.asarray(closed)
    _, distance, faces = model.nearest.on_surface(data[:, :3])
    touching = np.isclose(distance, data[:, 3] * 0.5, atol=1e-7, rtol=0)
    assert np.any(touching & (model.face_normals[faces, 2] < -0.7))


@pytest.mark.parametrize("embed", [0.0, 0.015])
def test_intentional_ceiling_attachment_does_not_allow_sidewall_penetration(embed):
    beads = _column()
    # The original tip clears this short wall, but a sphere lifted toward the
    # ceiling would cut into its lower left corner.
    wall = _box((0.1, 2.0, 0.34), (0.12, 0.0, 1.03))
    model = trimesh.util.concatenate([_ceiling(0.94), wall])
    _assert_no_intersections(beads, wall)

    closed, _ = _close(beads, model, [(0.0, 0.0, 0.94)], embed=embed)

    _assert_preserved_and_supported(beads, closed, model)
    _assert_no_intersections(closed, wall)


def test_contact_search_does_not_jump_through_an_intervening_shelf():
    beads = _column()
    intervening_shelf = _ceiling(0.84, thickness=0.02)
    model = trimesh.util.concatenate([intervening_shelf, _ceiling(0.98)])

    status = []
    closed, _ = _close(beads, model, [(0.0, 0.0, 0.98)], contact_status=status)

    _assert_preserved_and_supported(beads, closed, model)
    _assert_no_intersections(closed, intervening_shelf)
    # The lower solid blocks this column's route to the higher target.
    assert all(z < 0.84 for _, _, z, _ in closed)
    assert status == [False], "Touching the blocking shelf does not support the target above it"


@pytest.mark.parametrize("extra_budget", [0, 1, 2, 3, 6])
def test_ceiling_closure_respects_budget_without_leaving_partial_floating_chains(extra_budget):
    beads = _column()
    model = _ceiling(1.30)
    limit = len(beads) + extra_budget

    closed, _ = _close(beads, model, [(0.0, 0.0, 1.30)], max_beads=limit)

    assert len(closed) <= limit
    _assert_preserved_and_supported(beads, closed, model)
    _assert_no_intersections(closed, model)
    if extra_budget >= 3:
        _assert_flat_contact(closed, 1.30)


@pytest.mark.parametrize("repeated_targets", [
    [(0.0, 0.0, 1.30)] * 8,
    [(0.0, 0.0, 1.30), (0.01, 0.0, 1.30), (0.0, 0.01, 1.30)],
], ids=["identical-samples", "nearby-samples-on-the-same-tip"])
def test_repeated_contact_samples_share_one_ceiling_connection(repeated_targets):
    beads = _column()
    model = _ceiling(1.30)
    once, once_changed = _close(beads, model, [(0.0, 0.0, 1.30)])

    closed, changed = _close(beads, model, repeated_targets)

    assert closed == once
    assert changed == once_changed
    _assert_preserved_and_supported(beads, closed, model)
    _assert_flat_contact(closed, 1.30)


@pytest.mark.parametrize("diameter", [0.2, 0.4])
@pytest.mark.parametrize("embed", [0.0, 0.015])
def test_an_existing_surface_contact_needs_no_extra_bead_or_budget(diameter, embed):
    beads = _column(bottom=0.4 * diameter, diameter=diameter)
    ceiling_z = beads[-1][2] + diameter / 2
    status, positions = [], []
    closed, added, resolved = close_ceiling_gaps(
        beads, _ceiling(ceiling_z), [(0, 0, ceiling_z)], DIAMETER, 0.02,
        embed=embed, max_beads=len(beads), contact_status=status,
        contact_positions=positions)
    assert closed == beads
    assert added == 0
    assert status == [True]
    assert positions == [beads[-1][:3]]
    assert resolved == [beads[-1][2]]


def test_embed_allowance_never_counts_a_positive_air_gap_as_contact():
    beads = _column(bottom=0.016, diameter=0.04, count=29)
    status = []
    closed, _, _ = close_ceiling_gaps(
        beads, _ceiling(0.95), [(0, 0, 0.95)], DIAMETER, 0.02,
        embed=0.05, contact_status=status)
    if status[0]:
        assert any(z + d / 2 >= 0.95 - 1e-7 for _, _, z, d in closed)
    _assert_preserved_and_supported(beads, closed, _ceiling(0.95))


@pytest.mark.parametrize("clearance", [0.02, 0.8])
def test_roof_faces_behind_the_contact_do_not_apply_side_clearance(clearance):
    beads = _column()
    # A small, thin roof puts its opposite and side faces within XY clearance
    # of the contact. These lie behind the intended underside interface.
    model = _box((0.4, 0.4, 0.04), (0, 0, 0.96))
    closed, added, _ = close_ceiling_gaps(
        beads, model, [(0, 0, 0.94)], DIAMETER, clearance)
    assert added > 0
    _assert_preserved_and_supported(beads, closed, model)
    _assert_flat_contact(closed, 0.94)
    _assert_no_intersections(closed, model)


def test_full_resolved_position_tracks_a_contact_that_moves_sideways():
    angle = np.deg2rad(30)
    model = trimesh.creation.box(extents=(2, 2, 0.16))
    model.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    model.apply_translation([0, 0, 0.94 + 0.08 / np.cos(angle)])
    beads = _column()
    status, positions = [], []
    closed, _, heights = close_ceiling_gaps(
        beads, model, [(0, 0, 0.94)], DIAMETER, 0.02,
        contact_origins=[beads[-1][:3]], contact_status=status,
        contact_positions=positions)
    assert status == [True]
    assert abs(positions[0][0]) > 0.001
    assert positions[0] in [bead[:3] for bead in closed]
    assert heights == [positions[0][2]]
    _assert_preserved_and_supported(beads, closed, model)


def test_contact_status_remains_false_when_a_wall_blocks_attachment():
    beads = _column()
    wall = _box((0.1, 2.0, 0.34), (0.12, 0.0, 1.03))
    model = trimesh.util.concatenate([_ceiling(0.94), wall])
    status, positions = [], []
    closed, _, resolved = close_ceiling_gaps(
        beads, model, [(0, 0, 0.94)], DIAMETER, 0.02,
        embed=0.015, contact_status=status, contact_positions=positions)
    assert status == [False]
    assert positions == resolved == [None]
    _assert_no_intersections(closed, wall)
