# -*- coding: utf-8 -*-
"""로컬 웹 UI.

    pellet-support-web
    또는  python -m pellet_support.webapp

브라우저에서 http://127.0.0.1:5000 을 열고 모델 파일을 끌어다 놓으면
서포터가 포함된 파일을 되돌려준다. 파일은 임시 폴더에만 잠깐 저장되고
서버를 끄면 사라진다.

GitHub Pages 같은 정적 호스팅에서는 동작하지 않는다. trimesh / shapely 가
서버에서 실행되어야 하기 때문에 반드시 로컬(또는 직접 띄운 서버)에서 돌려야 한다.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import re
import shutil
import tempfile
import uuid
import warnings
import webbrowser
from typing import Dict, Optional

import trimesh
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from .params import SupportGenParams
from .autotune import auto_tune_bead_diameter
from .pipeline import generate_support, make_params
from .report import export_preview, load_mesh, measure_packing
from .validation import (
    InvalidModelError,
    InvalidParameterError,
    TooManyBeadsError,
)

MESH_EXTS = (".stl", ".obj", ".3mf", ".ply", ".off")
MAX_UPLOAD_MB = 200

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(Exception)
def handle_any_error(exc):
    """어떤 예외든 빈 응답이나 HTML 에러 페이지 대신 JSON으로 돌려준다.

    개별 라우트의 try/except 가 못 잡은 예외(werkzeug 레벨 에러, 업로드
    용량 초과 413 등)가 기본 HTML 에러 페이지로 나가면 프런트엔드의
    res.json() 이 'Unexpected end of JSON input' 으로 실패해서 원인을 알 수
    없는 상태가 된다. 이 핸들러가 그런 경우를 마지막으로 가로챈다.

    다만 이것도 프로세스 자체가 죽는 경우(메모리 부족으로 OS가 강제 종료)는
    막을 수 없다 — 그때는 서버를 실행 중인 터미널을 직접 확인해야 한다.
    """
    from werkzeug.exceptions import HTTPException

    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description or str(exc)), exc.code or 500
    app.logger.exception("처리되지 않은 예외")
    return jsonify(error=f"{type(exc).__name__}: {exc}"), 500

#: job_id -> 임시 폴더 경로
_JOBS: Dict[str, str] = {}
_JOBS_LOCK = threading.Lock()


def safe_filename(name: str) -> str:
    """경로 조작만 막고 한글 등 유니코드 파일명은 살린다.

    werkzeug 의 secure_filename 은 비ASCII 문자를 통째로 버려서
    '배_모델.3mf' 가 '_.3mf' 가 되어 버린다. 여기서는 디렉터리 구분자와
    제어문자, 윈도우 금지문자만 제거한다.
    """
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    return name[:120] or "model"


def _form_float(name: str, default: float) -> float:
    raw = request.form.get(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _form_optional_float(name: str) -> Optional[float]:
    """비어 있으면 None. make_params 의 자동 계산(노즐의 절반)을 쓰기 위함."""
    raw = request.form.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _form_int(name: str, default: int) -> int:
    raw = request.form.get(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _form_bool(name: str) -> bool:
    return request.form.get(name, "").lower() in ("1", "true", "on", "yes")


@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


#: 진행 중/완료된 작업 상태. job -> dict(status, payload, error, http_status)
_JOB_STATE: Dict[str, dict] = {}
_JOB_STATE_LOCK = threading.Lock()
#: 완료된 작업을 이 시간(초)이 지나면 정리한다(장시간 켜둔 서버에서
#: 무한정 쌓이지 않도록).
_JOB_TTL_S = 3600


def _run_generation_job(job, workdir, in_path, stem, ext, out_ext, form):
    """실제 계산 전체(파일 로딩~파일 저장~응답 구성)를 배경 스레드에서 돈다.

    예전에는 이 전체를 45초 안에 끝내야 했다. 45초를 넘기면 계산은
    계속되는데 사용자는 응답을 못 받아 "서버가 죽었다"로 보였다(나뭇가지
    거치대 실측 126초). 이제는 시간 제한 없이 끝까지 돌리고, 그동안 브라우저는
    /api/status 를 반복 조회해서 진행 상태만 확인한다 — 터미널(CLI)과
    동일하게 시간 제한이 없다.
    """
    def set_state(status, **kw):
        with _JOB_STATE_LOCK:
            _JOB_STATE[job] = {"status": status, "at": time.time(), **kw}

    try:
        set_state("running", stage="모델 로딩")
        mesh = load_mesh(in_path)
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

        contact, body = make_params(
            nozzle_diameter_mm=form["nozzle"],
            bead_diameter_mm=form["bead_diameter"],
            overlap=form["overlap"],
            lateral_overlap=form["lateral_overlap"],
            vertical_overlap=form["vertical_overlap"],
            body_bead_ratio=form["body_bead_ratio"],
            stagger_period=form["stagger_period"],
            straight_columns=form["straight_columns"],
        )
        gen = SupportGenParams(
            nozzle_diameter_mm=form["nozzle"],
            layer_height_mm=contact.layer_height_mm(),
            overhang_angle_deg=form["overhang_angle"],
            contact_z_gap_layers=form["z_gap_layers"],
            xy_clearance_mm=form["xy_clearance"],
            contact_layers=form["contact_layers"],
            min_bead_to_nozzle_ratio=form["min_bead_ratio"],
            allow_internal_supports=form["allow_internal_supports"],
            tree_enabled=form["tree_enabled"],
            fallback_solid=not form["no_fallback_solid"],
            max_layers=form["max_layers"],
        )

        tuning = None
        if form["auto_tune"] and form["bead_diameter"] is None:
            set_state("running", stage="구슬 크기 자동 선택")
            tuning = auto_tune_bead_diameter(
                mesh, gen, contact, target_fill=form["auto_target"])
            contact, body = make_params(
                nozzle_diameter_mm=form["nozzle"],
                bead_diameter_mm=tuning.chosen.bead_diameter_mm,
                overlap=form["overlap"],
                lateral_overlap=form["lateral_overlap"],
                vertical_overlap=form["vertical_overlap"],
                body_bead_ratio=form["body_bead_ratio"],
                stagger_period=form["stagger_period"],
                straight_columns=form["straight_columns"],
            )
            gen = dataclasses.replace(gen, layer_height_mm=contact.layer_height_mm())

        set_state("running",
                  stage="트리 골격 생성" if gen.tree_enabled else "서포터 생성")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = generate_support(
                mesh, gen, contact, body,
                detail=form["sphere_detail"], verbose=False,
            )
        notes = [str(w.message) for w in caught]

        if result.mesh.is_empty:
            set_state("error", error=(
                "이 모델에는 서포터가 필요하지 않습니다. "
                "오버행 각도를 올려서 다시 시도해 보세요."
            ), http_status=422)
            return

        set_state("running", stage="파일 저장")
        files = []
        support_name = f"{stem}_support{out_ext}"
        result.mesh.export(os.path.join(workdir, support_name))
        files.append({"name": support_name, "label": "서포터만"})

        if form["with_model"]:
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="model")
            scene.add_geometry(result.mesh, node_name="support")
            both_name = f"{stem}_with_support{out_ext}"
            scene.export(os.path.join(workdir, both_name))
            files.append({"name": both_name, "label": "모델 + 서포터"})

        preview_url = None
        try:
            candidates = [l["layer"] for l in result.plan.layers if l["beads"]]
            if candidates:
                export_preview(
                    result.plan, result.slices, candidates[-1],
                    os.path.join(workdir, "preview.png"),
                )
                preview_url = f"/files/{job}/preview.png"
        except Exception:
            pass  # matplotlib 이 없으면 미리보기만 건너뛴다

        stats = None
        if not gen.tree_enabled:
            try:
                stats = measure_packing(result.plan, gen, contact.pitch_mm())
            except Exception:
                pass  # scipy 가 없으면 검증만 건너뛴다

        n_beads = sum(len(l["beads"]) for l in result.plan.layers)
        payload = dict(
            job=job,
            files=[{**f, "url": f"/files/{job}/{f['name']}"} for f in files],
            preview=preview_url,
            bead_diameter=round(contact.bead_diameter_mm, 4),
            layer_height=round(gen.layer_height_mm, 4),
            pitch=round(contact.pitch_mm(), 4),
            layers=len(result.slices),
            beads=n_beads,
            triangles=len(result.mesh.faces),
            volume_cm3=round(abs(result.mesh.volume) / 1000.0, 3),
            measured=stats,
            notes=notes,
            auto_tuned=(None if tuning is None else {
                "bead_diameter": round(tuning.chosen.bead_diameter_mm, 3),
                "fillable": round(tuning.chosen.fillable_fraction, 3),
                "met_target": tuning.met_target,
            }),
        )
        set_state("done", payload=payload)
    except (InvalidParameterError, InvalidModelError) as exc:
        set_state("error", error=str(exc), http_status=400)
    except TooManyBeadsError as exc:
        set_state("error", error=str(exc), http_status=413)
    except ModuleNotFoundError as exc:
        set_state("error", http_status=500, error=(
            f"'{exc.name}' 패키지가 없습니다. "
            f"터미널에서 다음을 실행하세요:  pip install {exc.name}"))
    except MemoryError:
        set_state("error", http_status=500,
                  error="모델이 너무 큽니다. 노즐 지름을 키우거나 모델을 단순화하세요.")
    except Exception as exc:  # noqa: BLE001 - 사용자에게 원인을 그대로 보여준다
        set_state("error", http_status=500, error=f"{type(exc).__name__}: {exc}")


def _cleanup_old_jobs():
    now = time.time()
    with _JOB_STATE_LOCK:
        stale = [j for j, s in _JOB_STATE.items()
                if s.get("status") != "running" and now - s.get("at", now) > _JOB_TTL_S]
        for j in stale:
            del _JOB_STATE[j]


@app.route("/api/generate", methods=["POST"])
def api_generate():
    upload = request.files.get("model")
    if upload is None or not upload.filename:
        return jsonify(error="모델 파일을 선택하세요."), 400

    filename = safe_filename(upload.filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in MESH_EXTS:
        return jsonify(
            error=f"지원하지 않는 형식입니다: {ext or '확장자 없음'}. "
                  f"{', '.join(MESH_EXTS)} 중 하나를 올려주세요."
        ), 400

    job = uuid.uuid4().hex[:12]
    workdir = tempfile.mkdtemp(prefix=f"pellet-{job}-")
    with _JOBS_LOCK:
        _JOBS[job] = workdir

    in_path = os.path.join(workdir, filename)
    upload.save(in_path)

    out_ext = request.form.get("format", "same")
    if out_ext == "same" or out_ext not in MESH_EXTS:
        out_ext = ext

    # 무거운 작업에 필요한 폼 값을 전부 지금(요청 컨텍스트 안에서) 미리
    # 꺼내둔다. 배경 스레드에서는 request 객체를 못 쓴다(스레드 로컬이라
    # 요청이 끝나면 사라진다).
    form = dict(
        nozzle=_form_float("nozzle", 1.0),
        bead_diameter=_form_optional_float("bead_diameter"),
        overlap=_form_float("overlap", 0.08),
        lateral_overlap=_form_optional_float("lateral_overlap"),
        vertical_overlap=_form_optional_float("vertical_overlap"),
        body_bead_ratio=_form_float("body_bead_ratio", 0.97),
        stagger_period=_form_int("stagger_period", 3),
        straight_columns=_form_bool("straight_columns"),
        overhang_angle=_form_float("overhang_angle", 45.0),
        z_gap_layers=_form_int("z_gap_layers", 1),
        xy_clearance=_form_float("xy_clearance", 0.8),
        contact_layers=_form_int("contact_layers", 2),
        min_bead_ratio=_form_float("min_bead_ratio", 0.35),
        allow_internal_supports=_form_bool("allow_internal_supports"),
        tree_enabled=_form_bool("tree_enabled"),
        no_fallback_solid=_form_bool("no_fallback_solid"),
        max_layers=_form_int("max_layers", 4000),
        with_model=_form_bool("with_model"),
        auto_tune=_form_bool("auto_tune"),
        auto_target=_form_float("auto_target", 0.75),
        sphere_detail=_form_int("sphere_detail", 1),
    )

    with _JOB_STATE_LOCK:
        _JOB_STATE[job] = {"status": "running", "at": time.time(), "stage": "대기 중"}
    threading.Thread(
        target=_run_generation_job,
        args=(job, workdir, in_path, stem, ext, out_ext, form),
        daemon=True,
    ).start()
    _cleanup_old_jobs()

    # 202 Accepted: 작업을 받았고 지금 처리 중이라는 뜻. 결과는
    # /api/status/<job> 을 반복 조회해서 받는다 — 시간 제한이 없다.
    return jsonify(job=job, status="running"), 202


@app.route("/api/status/<job>")
def api_status(job):
    with _JOB_STATE_LOCK:
        state = _JOB_STATE.get(job)
    if state is None:
        return jsonify(error="알 수 없는 작업입니다. 다시 시도해 주세요."), 404
    if state["status"] == "running":
        return jsonify(status="running", stage=state.get("stage", "")), 200
    if state["status"] == "error":
        return jsonify(status="error", error=state.get("error", "알 수 없는 오류")), \
            state.get("http_status", 500)
    # done
    return jsonify(status="done", **state["payload"]), 200


@app.route("/files/<job>/<path:name>")
def files(job: str, name: str):
    with _JOBS_LOCK:
        workdir = _JOBS.get(job)
    if workdir is None:
        return jsonify(error="만료된 작업입니다. 다시 생성해 주세요."), 404
    as_attachment = not name.endswith(".png")
    return send_from_directory(workdir, name, as_attachment=as_attachment)


def cleanup() -> None:
    with _JOBS_LOCK:
        for path in _JOBS.values():
            shutil.rmtree(path, ignore_errors=True)
        _JOBS.clear()


def _running_in_container() -> bool:
    """GitHub Codespaces나 일반 devcontainer 안인지 감지한다.

    이 안에서 127.0.0.1(루프백)로만 열면 컨테이너 밖의 포트 포워딩 프록시가
    접속할 수 없어서, 터미널엔 'Running on ...'이 떠 있는데 브라우저에선
    포워딩 프록시가 404를 내는 상황이 된다. 반드시 0.0.0.0(모든 인터페이스)
    으로 열어야 프록시가 들어올 수 있다.
    """
    return (
        os.environ.get("CODESPACES") == "true"
        or os.environ.get("REMOTE_CONTAINERS") == "true"
        or os.path.exists("/.dockerenv")
    )


def main(argv=None) -> int:
    import argparse

    in_container = _running_in_container()

    ap = argparse.ArgumentParser(
        prog="pellet-support-web", description="펠릿 서포터 생성기 로컬 웹 UI"
    )
    ap.add_argument("--host", default="0.0.0.0" if in_container else "127.0.0.1",
                    help="바인딩 주소. Codespaces/devcontainer 안에서는 "
                         "자동으로 0.0.0.0 (외부 포트 포워딩이 접속 가능하도록)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", default=in_container,
                    help="브라우저 자동 실행 안 함 "
                         "(Codespaces/devcontainer 안에서는 기본적으로 켜짐)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    print(f"펠릿 서포터 생성기 실행 중 -> {url}")
    if in_container:
        print(f"Codespaces/devcontainer 환경이 감지됐습니다. 포트 {args.port} 알림 "
              f"팝업의 'Open in Browser' 를 누르거나, 터미널 옆 PORTS 탭에서 "
              f"해당 포트를 여세요. (127.0.0.1 이 아니라 0.0.0.0 으로 열려야 "
              f"포트 포워딩이 접속할 수 있습니다.)")
    print("종료하려면 Ctrl+C")
    if not args.no_browser and not args.debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        # threaded=True 가 반드시 필요하다. 서포터 생성은 배경 스레드에서
        # 돌지만, 진행 상황을 확인하는 /api/status 폴링 요청은 별도의 HTTP
        # 연결이다. threaded=True 없이는 Flask 개발 서버가 한 번에 연결
        # 하나만 처리해서, 계산이 끝날 때까지 폴링 요청 자체가 응답을
        # 못 받는다 — 폴링 인프라를 다 만들어도 무용지물이 된다(실측:
        # CPU 부하가 걸린 동안 폴링이 몇 초가 아니라 완전히 멈췄다).
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
