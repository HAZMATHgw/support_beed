"""Keep exact sliced geometry while avoiding repeated full-mesh processing."""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh.intersections import mesh_multiplane

from pellet_support import slicing


def _reference_slices(mesh, layer_height):
    z_min, z_max = mesh.bounds[:, 2]
    heights = (np.arange(int(np.ceil((z_max - z_min) / layer_height))) + 0.5) * layer_height
    lines, _, _ = mesh_multiplane(mesh, [0, 0, z_min], [0, 0, 1], heights)
    result = []
    for segments in lines:
        polygons = [p.buffer(0) for p in slicing.segments_to_polygons(segments)]
        polygons = [p for p in polygons if not p.is_empty]
        result.append(unary_union(polygons) if polygons else Polygon())
    return result, heights + z_min


def _segmented_box(z):
    mesh = trimesh.creation.box(extents=(2, 3, 1))
    mesh.apply_translation((0, 0, z + 0.5))
    return mesh


@pytest.mark.parametrize("mesh,layer_height", [
    (trimesh.creation.icosphere(subdivisions=2, radius=2), 0.23),
    (trimesh.creation.annulus(r_min=1, r_max=2, height=3, sections=32), 0.35),
    (trimesh.util.concatenate([_segmented_box(0), _segmented_box(1.5)]), 1.0),
], ids=["curved-faces", "interior-hole", "coplanar-bottom-and-top"])
@pytest.mark.parametrize("translation", [(0, 0, 0), (-13.25, 7.75, 12.3)])
def test_pruned_plane_intersections_match_full_mesh_geometry(mesh, layer_height, translation):
    mesh = mesh.copy()
    mesh.apply_translation(translation)
    expected, expected_heights = _reference_slices(mesh, layer_height)
    actual, actual_heights = slicing.slice_model(mesh, layer_height, 1000)
    assert np.array_equal(actual_heights, expected_heights)
    assert len(actual) == len(expected)
    for new, old in zip(actual, expected):
        assert new.equals_exact(old, tolerance=0.0)
        assert new.is_valid


def test_layers_only_classify_triangles_in_their_height_range(monkeypatch):
    mesh = trimesh.util.concatenate([_segmented_box(z) for z in (0, 10, 20)])
    original = slicing.mesh_plane
    inspected_face_counts = []

    def counted(mesh, normal, origin, **kwargs):
        local = kwargs["local_faces"]
        assert local is not None
        assert np.all(local[:-1] <= local[1:])  # preserve original face ordering
        inspected_face_counts.append(len(local))
        return original(mesh, normal, origin, **kwargs)

    monkeypatch.setattr(slicing, "mesh_plane", counted)
    slices, heights = slicing.slice_model(mesh, 0.25, 1000)
    assert len(heights) == 84
    assert len(inspected_face_counts) == 12  # empty levels skip intersection work
    assert all(count < len(mesh.faces) for count in inspected_face_counts)
    assert sum(not part.is_empty for part in slices) == 12


def test_endpoint_grouping_preserves_holes_islands_and_degenerate_edges():
    outer = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)
    hole = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
    island = np.array([[10, 0], [11, 0], [11, 1], [10, 1]], dtype=float)
    segments = np.concatenate([np.stack((ring, np.roll(ring, -1, axis=0)), axis=1)
                               for ring in (outer, hole, island)])
    # Endpoints that round to the same coordinates must still join. Duplicate
    # vertices use the original first representative, not the rounded value.
    segments[1, 0] += [1e-7, 0]
    segments = np.concatenate((segments, [[[20, 20], [20, 20]]]))
    polygons = slicing.segments_to_polygons(segments)
    result = unary_union(polygons)
    assert result.is_valid
    assert result.area == pytest.approx(33.0, abs=1e-6)
    assert len(polygons) == 2
    assert sum(len(p.interiors) for p in polygons) == 1


def test_empty_segments_and_layer_guard_remain_supported():
    assert slicing.segments_to_polygons(None) == []
    assert slicing.segments_to_polygons(np.empty((0, 2, 2))) == []
    with pytest.raises(RuntimeError, match="레이어 수가 너무 많습니다"):
        slicing.slice_model(_segmented_box(0), 0.1, 2)
