"""The existing polling endpoint exposes automatic tuning progress live."""

import io
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import trimesh

pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.compare_tree_support import make_model
from pellet_support import webapp


@pytest.fixture
def client(monkeypatch, tmp_path):
    def job_directory(prefix):
        path = tmp_path / prefix
        path.mkdir()
        return str(path)

    monkeypatch.setattr(webapp.tempfile, "mkdtemp", job_directory)
    monkeypatch.setattr(webapp, "_JOBS", {})
    monkeypatch.setattr(webapp, "_JOB_STATE", {})
    with webapp.app.test_client() as test_client:
        yield test_client


def submit(client, **fields):
    cube = trimesh.creation.box(extents=(4, 4, 4))
    response = client.post("/api/generate", data={
        "model": (io.BytesIO(cube.export(file_type="stl")), "cube.stl"),
        "auto_tune": "1", "sphere_detail": "0", **fields,
    })
    assert response.status_code == 202, response.get_json()
    return response.get_json()["job"]


def wait_for_completion(client, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/status/{job}")
        if response.get_json()["status"] != "running":
            return response
        time.sleep(0.01)
    pytest.fail(f"Job {job} did not finish")


def test_tuning_progress_is_visible_before_the_background_step_finishes(client, monkeypatch):
    slice_ready, candidate_ready, generation_ready = (threading.Event() for _ in range(3))
    finish_slicing, finish_tuning, finish_generation = (threading.Event() for _ in range(3))
    stages = ("자동 선택: 모델 단면 분석", "자동 선택: 후보 2/8 · 구슬 0.18mm 평가")
    captured = []

    def tune(mesh, gen, contact, target_fill, progress_callback):
        assert callable(progress_callback)
        progress_callback(stages[0])
        slice_ready.set()
        assert finish_slicing.wait(5)
        progress_callback(stages[1])
        candidate_ready.set()
        assert finish_tuning.wait(5)
        return SimpleNamespace(
            chosen=SimpleNamespace(bead_diameter_mm=0.18, fillable_fraction=0.9),
            met_target=True,
        )

    def generate(mesh, gen, contact, body, **kwargs):
        captured.append((gen.layer_height_mm, contact.bead_diameter_mm))
        kwargs["progress_callback"]("트리 가새 보강")
        generation_ready.set()
        assert finish_generation.wait(5)
        # A cube needs no support; the normal empty-result response ends this job.
        return SimpleNamespace(mesh=trimesh.Trimesh())

    monkeypatch.setattr(webapp, "auto_tune_bead_diameter", tune)
    monkeypatch.setattr(webapp, "generate_support", generate)
    job = submit(client)
    try:
        assert slice_ready.wait(5)
        first = client.get(f"/api/status/{job}").get_json()
        assert first == {"status": "running", "stage": stages[0]}
        finish_slicing.set()
        assert candidate_ready.wait(5)
        second = client.get(f"/api/status/{job}").get_json()
        assert second == {"status": "running", "stage": stages[1]}
        finish_tuning.set()
        assert generation_ready.wait(5)
        third = client.get(f"/api/status/{job}").get_json()
        assert third == {"status": "running", "stage": "트리 가새 보강"}
    finally:
        finish_slicing.set()
        finish_tuning.set()
        finish_generation.set()
        result = wait_for_completion(client, job)
    assert result.status_code == 422
    assert "서포터가 필요하지 않습니다" in result.get_json()["error"]
    assert captured[0][1] == pytest.approx(0.18)
    assert captured[0][0] == pytest.approx(0.18 * 0.92 * (2 / 3) ** 0.5)


def test_manual_bead_size_skips_tuning_even_if_automatic_selection_is_checked(client, monkeypatch):
    def unexpected_tuning(*args, **kwargs):
        raise AssertionError("An explicit bead diameter must skip automatic tuning")

    captured = []

    def generate(mesh, gen, contact, body, **kwargs):
        captured.append(contact.bead_diameter_mm)
        return SimpleNamespace(mesh=trimesh.Trimesh())

    monkeypatch.setattr(webapp, "auto_tune_bead_diameter", unexpected_tuning)
    monkeypatch.setattr(webapp, "generate_support", generate)
    job = submit(client, bead_diameter="0.2")
    response = wait_for_completion(client, job)
    assert response.status_code == 422
    assert captured == [0.2]


def test_tuning_failure_replaces_running_progress_with_a_terminal_error(client, monkeypatch):
    def failed_tuning(mesh, gen, contact, target_fill, progress_callback):
        progress_callback("자동 선택: 모델 단면 분석")
        raise RuntimeError("단면 분석을 완료하지 못했습니다")

    monkeypatch.setattr(webapp, "auto_tune_bead_diameter", failed_tuning)
    response = wait_for_completion(client, submit(client))
    assert response.status_code == 500
    payload = response.get_json()
    assert payload["status"] == "error"
    assert "단면 분석을 완료하지 못했습니다" in payload["error"]


def test_real_tree_reports_each_stage_before_the_expensive_geometry_work(monkeypatch):
    from pellet_support import pipeline, regions, skeleton
    from pellet_support.params import SupportGenParams

    stages = []
    entered = []

    def observing(original, message):
        def call(*args, **kwargs):
            assert stages and stages[-1] == message
            entered.append(message)
            return original(*args, **kwargs)
        return call

    operations = [
        (pipeline, "slice_model", "모델 단면 계산"),
        (regions, "detect_overhangs", "오버행 탐색"),
        (skeleton, "extract_contact_points", "트리 접점 선택"),
        (skeleton, "grow_branches", "트리 가지 성장·병합"),
        (skeleton, "skeleton_to_bead_seeds", "트리 구슬 배치"),
        (skeleton, "add_bracing", "트리 가새 보강"),
        (skeleton, "settle_collisions", "모델 충돌 보정"),
        (skeleton, "prune_floating", "떠 있는 구슬 정리"),
        (pipeline, "plan_to_mesh", "서포터 메쉬 생성"),
    ]
    for module, name, message in operations:
        monkeypatch.setattr(module, name, observing(getattr(module, name), message))

    contact, body = pipeline.make_params(nozzle_diameter_mm=2.0, bead_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=2.0, tree_enabled=True,
                           layer_height_mm=contact.layer_height_mm(),
                           detection_layer_height_mm=0.25, xy_clearance_mm=0.3,
                           contact_z_gap_mm=0.2, min_island_area_mm2=0.5)
    result = pipeline.generate_support(make_model("bridge"), gen, contact, body,
                                       detail=0, verbose=False, progress_callback=stages.append)
    assert not result.mesh.is_empty
    assert entered == [message for _, _, message in operations]
