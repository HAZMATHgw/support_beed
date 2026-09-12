"""Collision repair preserves physical landings without accepting nearby air."""

from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pellet_support.skeleton import fix_residual_collisions, settle_collisions


def floor_mesh():
    mesh = trimesh.creation.box(extents=(20, 20, 3))
    mesh.apply_translation((0, 0, 1.5))
    return mesh


@pytest.mark.parametrize("diameter, clearance", [(0.2, 0.8), (0.4, 0.1)])
def test_real_floor_contact_is_not_lifted_by_xy_clearance(diameter, clearance):
    floor = floor_mesh()
    seed = [(0.0, 0.0, 3.0 + diameter * 0.5, diameter)]
    repaired = fix_residual_collisions(seed, floor, clearance, z_gap=0.3)
    assert np.asarray(repaired) == pytest.approx(np.asarray(seed))
    settled, removed = settle_collisions(seed, floor, clearance, z_gap=0.3)
    assert removed == 0
    assert np.asarray(settled) == pytest.approx(np.asarray(seed))


def test_nearby_air_above_floor_does_not_count_as_physical_contact():
    seed = [(0.0, 0.0, 3.12, 0.2)]  # 0.02 mm above the floor.
    repaired = fix_residual_collisions(seed, floor_mesh(), 0.1, z_gap=0.3)
    assert repaired[0][2] > seed[0][2]


def test_upward_face_at_corner_does_not_anchor_a_sphere_beside_the_floor():
    floor = floor_mesh()
    # Exactly tangent to an edge from outside the footprint. The closest face
    # may be the upward-facing top, but its normal is not the contact direction.
    seed = [(10.16, 0.0, 3.12, 0.4)]
    _, distance, _ = trimesh.proximity.ProximityQuery(floor).on_surface(np.array([seed[0][:3]]))
    assert distance[0] == pytest.approx(0.2)
    repaired = fix_residual_collisions(seed, floor, 0.1, z_gap=0.3)
    assert np.linalg.norm(np.array(repaired[0][:3]) - np.array(seed[0][:3])) > 0.01


def test_ceiling_contact_still_respects_requested_z_gap():
    ceiling = trimesh.creation.box(extents=(20, 20, 1))
    ceiling.apply_translation((0, 0, 3.5))  # Underside at z = 3.
    seed = [(0.0, 0.0, 2.8, 0.4)]
    repaired = fix_residual_collisions(seed, ceiling, 0.8, z_gap=0.3)
    top = repaired[0][2] + repaired[0][3] * 0.5
    assert 3.0 - top >= 0.3 - 1e-6
    assert repaired[0][2] < seed[0][2]


def test_floor_penetration_still_gets_corrected():
    floor = floor_mesh()
    seed = [(0.0, 0.0, 3.1, 0.4)]
    repaired = fix_residual_collisions(seed, floor, 0.1, z_gap=0.3)
    assert repaired[0][2] - repaired[0][3] * 0.5 >= 3.0
