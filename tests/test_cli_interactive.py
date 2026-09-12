import os, sys, subprocess

def test_interactive_asks_nozzle_bead_tree_with_model(tmp_path):
    """대화형 모드로 노즐/구슬/트리/합본 여부를 물어봐야 한다."""
    inputs = "3\n1.5\ny\ny\n"
    out = str(tmp_path / "pytest_interactive.stl")
    result = subprocess.run(
        [sys.executable, "-m", "pellet_support.cli", "examples/out/table.stl", "--interactive",
         "-o", out],
        input=inputs, capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert "대화형 입력" in result.stdout
    assert f"합본 저장: {tmp_path / 'pytest_interactive_with_model.stl'}" in result.stdout


def test_with_model_file_follows_output_path_not_input_path(tmp_path):
    """-o 로 출력 위치를 지정하면 합본 파일도 그 위치를 따라야 한다.

    예전에는 항상 입력 파일 옆에 만들어서 -o 로 다른 폴더를 지정하면
    합본 파일만 엉뚱한 곳에 생겼다.
    """
    import os
    out = str(tmp_path / "pytest_wm_test.stl")
    result = subprocess.run(
        [sys.executable, "-m", "pellet_support.cli", "examples/out/table.stl", "--nozzle", "3",
         "--with-model", "-o", out],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out)
    assert (tmp_path / "pytest_wm_test_with_model.stl").exists()
