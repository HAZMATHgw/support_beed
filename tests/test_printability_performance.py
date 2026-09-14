"""Geometry-query regressions for bottom-up path repair."""

import numpy as np
import pytest
import trimesh

from pellet_support.printability import _outside_from_surface, repair_support_paths
from pellet_support.surface_query import CachedSurfaceQuery


@pytest.mark.parametrize("shape", ["box", "sphere", "concave", "open"])
def test_reused_nearest_result_matches_signed_distance_at_faces_and_edges(shape):
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=1)
    else:
        mesh = trimesh.creation.box()
        if shape == "concave":
            second = trimesh.creation.box()
            second.apply_translation([0.5, 0.5, 0.0])
            mesh = trimesh.util.concatenate([mesh, second])
        elif shape == "open":
            mesh.update_faces(np.arange(len(mesh.faces) - 2))
    random = np.random.default_rng(2718)
    points = random.uniform(-1.8, 1.8, (512, 3))
    # Include exact vertices, edges, and both sides of the surface tolerance.
    boundary = np.vstack((mesh.vertices, mesh.triangles_center,
                          mesh.triangles[:, :2].mean(axis=1)))
    points = np.vstack([points, boundary, boundary + 5e-8, boundary - 5e-8])
    closest, distance, faces = mesh.nearest.on_surface(points)
    cached = CachedSurfaceQuery(mesh).on_surface(points)
    for actual, expected in zip(cached, (closest, distance, faces)):
        np.testing.assert_array_equal(actual, expected)
    expected = mesh.nearest.signed_distance(points) <= 1e-7
    actual = _outside_from_surface(mesh, points, closest, distance, faces)
    np.testing.assert_array_equal(actual, expected)


def test_repair_does_not_repeat_surface_queries_inside_signed_distance(monkeypatch):
    model = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    model.apply_translation((0.0, 0.0, 0.5))
    beads = [(0.0, 0.0, 1.4 + i * 0.16, 0.2) for i in range(5)]
    original = model.nearest.on_surface
    cached_original = CachedSurfaceQuery.on_surface
    calls = []
    cached_calls = []

    def on_surface(points):
        calls.append(len(points))
        return original(points)

    def repeated_nearest_query(*args, **kwargs):
        pytest.fail("signed_distance repeats the already completed nearest query")

    def cached_on_surface(self, points):
        cached_calls.append(len(points))
        return cached_original(self, points)

    monkeypatch.setattr(model.nearest, "on_surface", on_surface)
    monkeypatch.setattr(model.nearest, "signed_distance", repeated_nearest_query)
    monkeypatch.setattr(CachedSurfaceQuery, "on_surface", cached_on_surface)
    repaired, added = repair_support_paths(beads, model, 0.0, 0.2, 0.05)
    assert added == 2
    assert repaired[-1] == (0.0, 0.0, 1.1, 0.2)
    # One finite-triangle landing check, then one batch for the whole path.
    assert calls == [1]
    assert cached_calls == [2]


def test_face_interior_sign_does_not_need_containment_rays(monkeypatch):
    model = trimesh.creation.box()
    points = np.array([[0.1, 0.2, 0.8], [0.1, 0.2, 0.2]])
    closest, distance, faces = model.nearest.on_surface(points)

    def unexpected_query(*args, **kwargs):
        pytest.fail("face-interior projection already determines the sign")

    monkeypatch.setattr(model.nearest, "on_surface", unexpected_query)
    monkeypatch.setattr(model.ray, "contains_points", unexpected_query)
    np.testing.assert_array_equal(
        _outside_from_surface(model, points, closest, distance, faces), [True, False])


def test_small_surface_batches_build_the_vertex_index_once(monkeypatch):
    import pellet_support.surface_query as module

    mesh = trimesh.creation.icosphere(subdivisions=2)
    original = module.cKDTree
    builds = []

    def build(vertices):
        builds.append(len(vertices))
        return original(vertices)

    monkeypatch.setattr(module, "cKDTree", build)
    query = CachedSurfaceQuery(mesh)
    for x in np.linspace(-2.0, 2.0, 20):
        points = np.array([[x, 0.1, 0.2]])
        actual = query.on_surface(points)
        expected = mesh.nearest.on_surface(points)
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)
    assert builds == [len(mesh.vertices)]


def test_surface_query_fallback_preserves_exact_candidates(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    query = CachedSurfaceQuery(mesh)
    points = np.random.default_rng(314).uniform(-1.5, 1.5, (64, 3))

    def batch_unavailable(*args, **kwargs):
        raise AttributeError("older Rtree without a batch intersection API")

    monkeypatch.setattr(mesh.triangles_tree, "intersection_v", batch_unavailable)
    for actual, expected in zip(query.on_surface(points), mesh.nearest.on_surface(points)):
        np.testing.assert_array_equal(actual, expected)
