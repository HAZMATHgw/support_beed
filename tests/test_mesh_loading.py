"""Mesh loading preserves geometry while repairing small open seams."""

from io import BytesIO
import zipfile

import numpy as np
import pytest
import trimesh

from pellet_support.report import load_mesh


def test_weld_closes_seam_across_decimal_rounding_boundary(tmp_path):
    # Separate triangles have 0.0002 mm cracks, straddling the old 0.001
    # rounding boundary at every original vertex.
    box = trimesh.creation.box()
    vertices = box.triangles.reshape(-1, 3).copy()
    vertices += 0.0005
    vertices += np.tile(np.array([-0.0001, 0.0001, -0.0001]), 12)[:, None]
    cracked = trimesh.Trimesh(vertices, np.arange(36).reshape(-1, 3), process=False)
    path = tmp_path / "cracked.ply"
    cracked.export(path)
    result = load_mesh(str(path))
    assert result.is_watertight
    assert len(result.vertices) == 8
    assert len(result.faces) == 12


def test_closed_thin_wall_is_not_collapsed_by_welding(tmp_path):
    box = trimesh.creation.box(extents=[1.0, 1.0, 0.0004])
    path = tmp_path / "thin.ply"
    box.export(path)
    result = load_mesh(str(path))
    assert result.is_watertight
    assert result.extents == pytest.approx(box.extents, rel=1e-6)
    assert len(result.faces) == 12


def test_welding_does_not_close_a_real_open_gap(tmp_path):
    left = trimesh.creation.box()
    right = trimesh.creation.box()
    right.apply_translation([1.002, 0, 0])
    # Leave one face open so the loader actually enters boundary repair.
    left.update_faces(np.arange(11))
    mesh = trimesh.util.concatenate([left, right])
    path = tmp_path / "gap.ply"
    mesh.export(path)
    result = load_mesh(str(path))
    assert len(result.split(only_watertight=False)) == 2
    assert len(result.vertices) == 16


def test_3mf_instances_keep_transforms_and_convert_declared_units(tmp_path):
    box = trimesh.creation.box()
    scene = trimesh.Scene()
    transform = trimesh.transformations.translation_matrix([2, 3, 4])
    scene.add_geometry(box, transform=transform)
    transform2 = trimesh.transformations.translation_matrix([5, 3, 4])
    scene.add_geometry(box, transform=transform2)
    archive = BytesIO(scene.export(file_type="3mf"))
    path = tmp_path / "inches.3mf"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(path, "w") as output:
        for entry in original.infolist():
            data = original.read(entry.filename)
            if entry.filename.lower().endswith("3dmodel.model"):
                data = data.replace(b'unit="millimeter"', b'unit="inch"')
                assert b'unit="inch"' in data
            output.writestr(entry, data)
    result = load_mesh(str(path))
    assert result.units == "mm"
    assert result.bounds == pytest.approx(
        np.array([[1.5, 2.5, 3.5], [5.5, 3.5, 4.5]]) * 25.4)
    assert len(result.split()) == 2


def test_mesh_without_valid_triangles_reports_load_error(tmp_path):
    mesh = trimesh.Trimesh(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], process=False)
    path = tmp_path / "degenerate.ply"
    mesh.export(path)
    with pytest.raises(RuntimeError, match="유효한 삼각형"):
        load_mesh(str(path))
