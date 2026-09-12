# -*- coding: utf-8 -*-
"""로컬 웹 UI 엔드포인트 검증."""

import io
import os

import pytest
import trimesh

flask = pytest.importorskip("flask")

from pellet_support import webapp  # noqa: E402


@pytest.fixture
def client():
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        yield c
    webapp.cleanup()


def _table_stl_bytes() -> bytes:
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = trimesh.creation.box(extents=[5, 5, 12])
            leg.apply_translation([sx * 10, sy * 6, 6])
            parts.append(leg)
    top = trimesh.creation.box(extents=[30, 20, 4])
    top.apply_translation([0, 0, 14])
    parts.append(top)
    mesh = trimesh.util.concatenate(parts)
    return mesh.export(file_type="stl")


def _submit_and_wait(client, data, timeout_s=60):
    """제출(202) 후 완료될 때까지 /api/status 를 반복 조회하는 테스트 헬퍼.

    API 가 동기식 단일 응답에서 '제출 후 폴링'으로 바뀌었다(웹 요청이 45초
    넘게 걸리면 연결이 끊기는 문제를 근본적으로 없애기 위해 — 나뭇가지
    거치대는 실측 94~126초가 걸렸는데, 이제는 시간 제한 없이 끝까지 돈다).
    """
    import time

    res = client.post("/api/generate", data=data,
                      content_type="multipart/form-data")
    body = res.get_json()
    if res.status_code != 202:
        return res.status_code, body  # 제출 자체가 거부된 경우(400 등)
    job = body["job"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/status/{job}")
        j = r.get_json()
        if j.get("status") != "running":
            return r.status_code, j
        time.sleep(0.2)
    raise TimeoutError(f"{timeout_s}초 안에 끝나지 않음: {job}")


def test_index_serves_the_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "펠릿 서포터 생성기" in res.get_data(as_text=True)


def test_generate_returns_downloadable_files(client):
    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "table.stl"),
        "nozzle": "3.0",  # 노즐 1.0(구슬 0.5mm)은 이 모델에서 구슬이
                          # 258,819개나 나와 131초가 걸린다. 3.0(구슬
                          # 1.5mm)이면 6초면 끝나면서 배위수도 이상적인
                          # 12 를 유지한다(5.0 은 더 빠르지만 9로 떨어진다).
        "overlap": "0.08",
        "tree_enabled": "0",  # This test measures lattice coordination.
        "sphere_detail": "0",
        "with_model": "1",
    }
    status, payload = _submit_and_wait(client, data)
    assert status == 200, payload
    assert payload["status"] == "done"

    assert payload["beads"] > 100
    # 기본 구슬 지름은 이제 노즐의 절반이다(노즐=bead 시절 값이 아님).
    expected_pitch = (3.0 * 0.5) * (1 - 0.08)
    assert payload["bead_diameter"] == pytest.approx(1.5, abs=1e-3)
    assert payload["layer_height"] == pytest.approx(expected_pitch * 0.81649658, abs=1e-3)
    # 배위수는 모델 크기 대비 구슬 크기에 민감하다(가장자리 효과). 여기선
    # 속도를 위해 구슬을 키웠으므로 이상적인 12 대신 '잘 다져졌다'는
    # 정도만 확인한다.
    assert payload["measured"]["median_coordination"] >= 8
    assert len(payload["files"]) == 2

    for f in payload["files"]:
        dl = client.get(f["url"])
        assert dl.status_code == 200
        assert len(dl.data) > 1000


def test_rejects_unsupported_extension(client):
    data = {"model": (io.BytesIO(b"not a mesh"), "notes.txt")}
    res = client.post("/api/generate", data=data,
                      content_type="multipart/form-data")
    assert res.status_code == 400
    assert "지원하지 않는 형식" in res.get_json()["error"]


def test_rejects_missing_file(client):
    res = client.post("/api/generate", data={},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_model_without_overhang_gets_a_clear_message(client):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    data = {
        "model": (io.BytesIO(cube.export(file_type="stl")), "cube.stl"),
        "sphere_detail": "0",
    }
    status, body = _submit_and_wait(client, data)
    assert status == 422
    assert "서포터가 필요하지 않습니다" in body["error"]


def test_expired_job_is_reported(client):
    res = client.get("/files/deadbeef/whatever.stl")
    assert res.status_code == 404


def test_unknown_job_status_returns_404(client):
    """존재하지 않는(혹은 만료된) 작업 번호를 조회하면 404 여야 한다."""
    res = client.get("/api/status/deadbeef00000")
    assert res.status_code == 404


def test_safe_filename_keeps_unicode_but_blocks_traversal():
    """한글 파일명은 살리고 경로 조작만 막는다."""
    from pellet_support.webapp import safe_filename

    assert safe_filename("배_모델.3mf") == "배_모델.3mf"
    assert safe_filename("../../etc/passwd.stl") == "passwd.stl"
    assert safe_filename("a<b>c:d.stl") == "a_b_c_d.stl"
    assert safe_filename("") == "model"
    assert "/" not in safe_filename("dir/sub/x.stl")


def test_korean_filename_survives_round_trip(client):
    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "배_테이블.stl"),
        "nozzle": "5.0",  # 기본값(1.0)은 이 모델에서 131초 걸린다.
        "sphere_detail": "0",
    }
    status, body = _submit_and_wait(client, data)
    assert status == 200
    name = body["files"][0]["name"]
    assert name == "배_테이블_support.stl", name


def test_unknown_route_still_returns_json(client):
    """전역 에러 핸들러 덕분에 404도 HTML 이 아니라 JSON으로 나온다."""
    res = client.get("/no-such-route")
    assert res.status_code == 404
    assert res.is_json
    assert "error" in res.get_json()


def test_oversized_upload_returns_json_not_html(client):
    """413(용량 초과)도 기본 HTML 에러 페이지 대신 JSON으로 나온다."""
    import io

    webapp.app.config["MAX_CONTENT_LENGTH"] = 100  # 테스트용으로 아주 작게
    try:
        data = {"model": (io.BytesIO(b"x" * 1000), "big.stl")}
        res = client.post("/api/generate", data=data,
                          content_type="multipart/form-data")
        assert res.status_code == 413
        assert res.is_json
    finally:
        webapp.app.config["MAX_CONTENT_LENGTH"] = None


def test_invalid_parameters_return_400_not_500(client):
    """사용자 입력 오류는 서버 오류(500)가 아니라 400 이어야 한다.

    검증(make_params/SupportGenParams)이 이제 배경 작업 안에서 일어나므로
    제출 자체는 202 로 받아들여지고, 실제 오류는 폴링 결과에 담긴다.
    """
    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "t.stl"),
        "sphere_detail": "0",
        "overlap": "1.5",
    }
    status, body = _submit_and_wait(client, data)
    assert status == 400
    assert "겹침" in body["error"]
    # 클래스 이름이 그대로 노출되면 안 된다
    assert "InvalidParameterError" not in body["error"]


def test_negative_clearance_is_rejected_by_api(client):
    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "t.stl"),
        "sphere_detail": "0",
        "xy_clearance": "-5",
    }
    status, body = _submit_and_wait(client, data)
    assert status == 400
    assert "XY 여유" in body["error"]


def test_submit_returns_immediately_regardless_of_job_duration(client):
    """제출(POST /api/generate)은 계산이 얼마나 오래 걸리든 즉시 202 로
    응답해야 한다. 예전에는 요청 하나가 45초 안에 안 끝나면 504 로 끊었는데
    (나뭇가지 거치대 실측 94~126초), 이제는 제출과 계산을 분리해서 시간
    제한 자체가 없다 — 터미널(CLI)과 동일한 대기 방식이다.
    """
    import time

    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "table.stl"),
        "nozzle": "1.0",  # 일부러 느린 설정(구슬 0.5mm, 258,819개, ~100초+)
        "sphere_detail": "0",
    }
    t0 = time.time()
    res = client.post("/api/generate", data=data,
                      content_type="multipart/form-data")
    elapsed = time.time() - t0
    assert res.status_code == 202
    assert elapsed < 5, f"제출 응답이 {elapsed:.1f}초나 걸렸다(즉시여야 함)"
    body = res.get_json()
    assert body["status"] == "running"
    assert "job" in body


def test_status_polling_reports_progress_stage(client):
    """진행 중에는 현재 단계(stage) 정보를 함께 줘야 한다."""
    import time

    data = {
        "model": (io.BytesIO(_table_stl_bytes()), "table.stl"),
        "nozzle": "1.0",
        "sphere_detail": "0",
    }
    res = client.post("/api/generate", data=data,
                      content_type="multipart/form-data")
    job = res.get_json()["job"]
    time.sleep(0.3)
    poll = client.get(f"/api/status/{job}")
    assert poll.status_code == 200
    j = poll.get_json()
    assert j["status"] in ("running", "done")
    if j["status"] == "running":
        assert "stage" in j
