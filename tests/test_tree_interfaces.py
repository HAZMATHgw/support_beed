"""CLI and real asynchronous web generation regressions for sparse trees."""

import io
from pathlib import Path
import sys
import time
import warnings

import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pellet_support import cli, webapp


@pytest.fixture
def bridge_stl():
    """A small bridge with an open, unsupported underside, in millimetres."""
    parts = []
    for size, centre in [
        ((6, 4, 0.5), (0, 0, 3.25)),
        ((0.75, 4, 3), (-2.625, 0, 1.5)),
        ((0.75, 4, 3), (2.625, 0, 1.5)),
    ]:
        part = trimesh.creation.box(extents=size)
        part.apply_translation(centre)
        parts.append(part)
    return trimesh.util.concatenate(parts).export(file_type="stl")


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Job files and state stay inside this test's temporary directory.
    def job_directory(prefix):
        directory = tmp_path / prefix
        directory.mkdir()
        return str(directory)

    monkeypatch.setattr(webapp.tempfile, "mkdtemp", job_directory)
    monkeypatch.setattr(webapp, "_JOBS", {})
    monkeypatch.setattr(webapp, "_JOB_STATE", {})
    with webapp.app.test_client() as test_client:
        yield test_client


def submit_and_wait(client, bridge_stl, **fields):
    submitted = client.post("/api/generate", data={
        "model": (io.BytesIO(bridge_stl), "bridge.stl"),
        "sphere_detail": "0",
        "auto_tune": "0",
        **fields,
    })
    assert submitted.status_code == 202, submitted.get_json()
    job = submitted.get_json()["job"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f"/api/status/{job}")
        payload = status.get_json()
        if payload["status"] != "running":
            return status.status_code, payload
        time.sleep(0.02)
    pytest.fail(f"Generation job {job} did not complete within 30 seconds")


@pytest.mark.parametrize("flags, expected", [
    ([], True), (["--tree"], True), (["--no-tree"], False), (["--grid"], False),
])
def test_cli_mode_and_interactive_default_respect_explicit_choice(
        flags, expected, monkeypatch, capsys):
    args = cli.build_parser().parse_args(["bridge.stl", *flags])
    assert args.nozzle == 0.4
    assert args.tree is expected
    monkeypatch.setattr("builtins.input", lambda _: "")
    cli._run_interactive_prompts(args)
    assert args.tree is expected


def test_cli_tree_controls_and_mutually_exclusive_modes(capsys):
    parser = cli.build_parser()
    args = parser.parse_args([
        "bridge.stl", "--tree-contact-spacing", "1.4",
        "--branch-angle", "30", "--branch-merge-distance", "2.6",
    ])
    assert (args.tree_contact_spacing, args.branch_angle,
            args.branch_merge_distance) == (1.4, 30.0, 2.6)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["bridge.stl", "--tree", "--grid"])
    assert error.value.code == 2


@pytest.mark.parametrize("fields, expected", [
    ({}, (None, 25.0, None)),
    ({"tree_contact_spacing": "1.4", "branch_angle": "30",
      "branch_merge_distance": "2.6"}, (1.4, 30.0, 2.6)),
])
def test_web_generates_downloadable_tree_and_exposes_settings_and_warnings(
        client, bridge_stl, monkeypatch, fields, expected):
    generated_params = []
    real_generate = webapp.generate_support
    warning_message = "Tree integration warning: check support contact spacing."

    def observed_generate(mesh, gen, contact, body, **kwargs):
        generated_params.append(gen)
        warnings.warn(warning_message, UserWarning)
        return real_generate(mesh, gen, contact, body, **kwargs)

    monkeypatch.setattr(webapp, "generate_support", observed_generate)
    status, payload = submit_and_wait(client, bridge_stl, **fields)
    assert status == 200, payload
    assert payload["status"] == "done"
    assert payload["tree_enabled"] is True
    assert payload["beads"] > 0
    assert payload["triangles"] > 0
    assert payload["bead_diameter"] == pytest.approx(0.2)
    assert warning_message in payload["notes"]
    gen = generated_params[0]
    assert gen.nozzle_diameter_mm == pytest.approx(0.4)
    assert (gen.tree_contact_spacing_mm, gen.branch_angle_deg,
            gen.branch_merge_distance_mm) == expected
    assert payload["tree_contact_spacing_mm"] == pytest.approx(expected[0] or 0.8)
    assert payload["branch_angle_deg"] == expected[1]
    assert payload["branch_merge_distance_mm"] == pytest.approx(expected[2] or 1.2)
    download = client.get(payload["files"][0]["url"])
    assert download.status_code == 200
    exported = trimesh.load(io.BytesIO(download.data), file_type="stl")
    assert len(exported.faces) == payload["triangles"]
    page = client.get("/").get_data(as_text=True)
    assert 'id="nozzle" value="0.4"' in page
    assert 'id="tree_enabled" checked' in page
    assert 'id="r-warn"' in page and "data.notes.map" in page


@pytest.mark.parametrize("field, value", [
    (field, value)
    for field in ("tree_contact_spacing", "branch_merge_distance")
    for value in ("0", "-1", "nan", "inf", "-inf")
] + [("branch_angle", value) for value in ("-1", "90", "nan", "inf", "-inf")])
def test_web_invalid_tree_numbers_finish_as_input_errors(client, bridge_stl, field, value):
    status, payload = submit_and_wait(client, bridge_stl, **{field: value})
    assert status == 400, payload
    assert payload["status"] == "error"
    assert payload["error"]


@pytest.mark.parametrize("field", [
    "tree_contact_spacing", "branch_angle", "branch_merge_distance",
])
def test_web_rejects_non_numeric_tree_settings_before_starting_job(client, bridge_stl, field):
    response = client.post("/api/generate", data={
        "model": (io.BytesIO(bridge_stl), "bridge.stl"), field: "not a number",
    })
    assert response.status_code == 400
    assert response.get_json()["error"]
    assert not webapp._JOB_STATE


def test_web_explicit_grid_selection_survives_form_parsing(client, bridge_stl, monkeypatch):
    submitted_forms = []

    class CapturedJob:
        def __init__(self, *, target, args, daemon):
            submitted_forms.append(args[-1])

        def start(self):
            pass

    monkeypatch.setattr(webapp.threading, "Thread", CapturedJob)
    response = client.post("/api/generate", data={
        "model": (io.BytesIO(bridge_stl), "bridge.stl"), "tree_enabled": "0",
    })
    assert response.status_code == 202
    assert submitted_forms[0]["tree_enabled"] is False
