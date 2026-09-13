"""Keep tuning geometry shared, observable, and honest about failures.

Operation counts express the performance contract without timing thresholds or
a large mesh fixture. Candidate scoring still uses real Shapely geometry.
"""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from shapely.geometry import Polygon, box
import trimesh

from pellet_support import SupportGenParams, make_params
from pellet_support import autotune


@pytest.fixture
def tuning_inputs():
    model = trimesh.creation.box(extents=[8, 8, 3])
    model.apply_translation([0, 0, 1.5])
    contact, _ = make_params(nozzle_diameter_mm=0.4)
    gen = SupportGenParams(nozzle_diameter_mm=0.4,
                           layer_height_mm=contact.layer_height_mm())
    return model, gen, contact


@pytest.fixture
def geometry_spies(monkeypatch):
    events = []
    slices = [box(-1, -1, 1, 1)] * 3
    support = [box(-4, -4, 4, 4).difference(box(-1, -1, 1, 1))] * 3
    overhang = [Polygon(), box(-3, -3, 3, 3), Polygon()]

    def slice_once(mesh, height, max_layers):
        events.append(("slice", height))
        return slices, np.array([0.5, 1.5, 2.5]) * height

    def build_once(*args, **kwargs):
        events.append(("build", None))
        return support, [Polygon()] * len(support)

    def detect_once(*args, **kwargs):
        events.append(("overhang", None))
        return overhang

    original_score = autotune.fillable_fraction

    def score(regions, diameter):
        events.append(("candidate", diameter))
        return original_score(regions, diameter)

    spies = dict(events=events, slice=Mock(side_effect=slice_once),
                 build=Mock(side_effect=build_once), detect=Mock(side_effect=detect_once),
                 score=Mock(side_effect=score), overhang=overhang)
    monkeypatch.setattr(autotune, "slice_model", spies["slice"])
    monkeypatch.setattr(autotune, "build_support_regions", spies["build"])
    monkeypatch.setattr(autotune, "detect_overhangs", spies["detect"], raising=False)
    monkeypatch.setattr(autotune, "fillable_fraction", spies["score"])
    return spies


@pytest.mark.parametrize("steps", [3, 8, 20])
@pytest.mark.parametrize("tree", [False, True], ids=["grid", "tree"])
def test_candidates_share_one_geometry_pass(tuning_inputs, geometry_spies, steps, tree):
    model, gen, contact = tuning_inputs
    # 트리 상한은 오버행의 평균 '높이'에서 정해진다. 이 모델은 3mm 밖에
    # 안 되는 얇은 상자라 기본 비율(0.1)로는 상한이 인쇄 가능 최소보다도
    # 작아져 후보가 하나로 뭉개진다. 여기서 검증하려는 건 비율 자체가
    # 아니라 "형상 탐지를 한 번만 공유하는지"이므로, 비율을 넉넉히 올려
    # steps 개만큼 실제로 퍼진 후보가 나오게 한다.
    gen = replace(gen, auto_tree_height_ratio=10.0) if tree else gen
    result = autotune.auto_tune_bead_diameter(model, replace(gen, tree_enabled=tree),
                                             contact, steps=steps)
    assert len(result.candidates) == steps
    assert geometry_spies["score"].call_count == steps
    assert geometry_spies["slice"].call_count == 1
    assert geometry_spies["build"].call_count == (0 if tree else 1)
    assert geometry_spies["detect"].call_count == (1 if tree else 0)
    assert result.chosen.estimated_beads > 0
    if tree:
        expected_area = sum(region.area for region in geometry_spies["overhang"])
        for call in geometry_spies["score"].call_args_list:
            assert sum(region.area for region in call.args[0]) == pytest.approx(expected_area)


@pytest.mark.parametrize("tree", [False, True], ids=["grid", "tree"])
def test_equal_minimum_and_maximum_diameters_are_scored_once(tuning_inputs, geometry_spies, tree):
    model, gen, contact = tuning_inputs
    # 구슬 상한은 이제 빈 공간(오버행 단면) 크기에서 정해지므로, 최소치와
    # 확실히 같아지게 하려면 인쇄 가능 최소를 그 상한보다 훨씬 크게 준다.
    # max(min_bead, void_based) 는 그러면 항상 min_bead 로 고정된다.
    gen = replace(gen, tree_enabled=tree, min_bead_diameter_mm=5.0)
    result = autotune.auto_tune_bead_diameter(model, gen, contact, steps=20)
    assert [candidate.bead_diameter_mm for candidate in result.candidates] == [5.0]
    assert result.chosen.bead_diameter_mm == 5.0
    assert geometry_spies["slice"].call_count == 1
    assert geometry_spies["score"].call_count == 1


@pytest.mark.parametrize("tree", [False, True], ids=["grid", "tree"])
def test_progress_reports_each_phase_before_expensive_work(tuning_inputs, geometry_spies, tree):
    model, gen, contact = tuning_inputs
    events = geometry_spies["events"]

    def progress(message):
        assert isinstance(message, str) and message.strip()
        events.append(("progress", message))

    autotune.auto_tune_bead_diameter(model, replace(gen, tree_enabled=tree), contact,
                                     steps=3, progress_callback=progress)
    phase_names = ["slice", "overhang" if tree else "build", "candidate"]
    phase_indices = [next(i for i, event in enumerate(events) if event[0] == phase)
                     for phase in phase_names]
    previous = -1
    for phase_index in phase_indices:
        assert any(kind == "progress" for kind, _ in events[previous + 1:phase_index])
        previous = phase_index
    messages = [message for kind, message in events if kind == "progress"]
    assert len(set(messages)) >= 3


@pytest.mark.parametrize("tree, failed_operation", [
    (False, "slice"), (False, "build"), (True, "slice"), (True, "detect"),
])
def test_geometry_errors_keep_the_original_cause(tuning_inputs, geometry_spies, tree, failed_operation):
    model, gen, contact = tuning_inputs
    cause = RuntimeError(f"geometry backend failure during {failed_operation}")
    geometry_spies[failed_operation].side_effect = cause
    with pytest.raises(Exception) as caught:
        autotune.auto_tune_bead_diameter(model, replace(gen, tree_enabled=tree),
                                         contact, steps=8)
    assert caught.value is cause or caught.value.__cause__ is cause
    assert geometry_spies[failed_operation].call_count == 1
    assert geometry_spies["score"].call_count == 0


def test_explicit_detection_height_is_honoured(tuning_inputs, geometry_spies):
    model, gen, contact = tuning_inputs
    gen = replace(gen, detection_layer_height_mm=0.3)
    autotune.auto_tune_bead_diameter(model, gen, contact)
    assert geometry_spies["slice"].call_count == 1
    assert geometry_spies["slice"].call_args.args[1] == pytest.approx(0.3)


def test_default_detection_height_stays_at_the_base_profile(tuning_inputs, geometry_spies):
    model, gen, contact = tuning_inputs
    autotune.auto_tune_bead_diameter(model, gen, contact, steps=8)
    assert geometry_spies["slice"].call_count == 1
    assert geometry_spies["slice"].call_args.args[1] == pytest.approx(min(0.4, gen.layer_height_mm))


def test_detection_height_respects_the_maximum_layer_count(tuning_inputs, geometry_spies):
    model, gen, contact = tuning_inputs
    gen = replace(gen, detection_layer_height_mm=0.1, max_detection_layers=4)
    autotune.auto_tune_bead_diameter(model, gen, contact)
    assert geometry_spies["slice"].call_count == 1
    assert geometry_spies["slice"].call_args.args[1] == pytest.approx(model.extents[2] / 4)


def test_tree_target_cannot_be_met_by_a_nonexistent_filler_bonus(tuning_inputs, geometry_spies):
    model, gen, contact = tuning_inputs
    # 이 모델은 3mm 밖에 안 되는 얇은 상자라, 기본 높이 비율(0.1)로는 상한이
    # 인쇄 가능 최소로 무너져 largest 가 사실상 아무것도 침식하지 않는다.
    # 비율을 올려 largest 가 1.46mm 사각형을 실제로 좁힐 만큼 커지게 한다.
    gen = replace(gen, tree_enabled=True, auto_tree_height_ratio=3.0)
    geometry_spies["detect"].side_effect = lambda *args, **kwargs: [
        Polygon(), box(0, 0, 1.46, 1.46), Polygon()
    ]
    probe = autotune.auto_tune_bead_diameter(model, gen, contact, target_fill=0.0, steps=2)
    largest = max(probe.candidates, key=lambda candidate: candidate.bead_diameter_mm)
    smaller = min(probe.candidates, key=lambda candidate: candidate.bead_diameter_mm)
    # 구슬 상한이 이제 빈 공간(여기서는 1.46x1.46 오버행 사각형) 크기에서
    # 정해지므로 0.2mm 로 고정되어 있지 않다. largest 가 실제로 고른 지름을
    # 그대로 침식 공식에 대입해 같은 기하 계산이 여전히 맞는지 확인한다.
    assert largest.fillable_fraction == pytest.approx(
        ((1.46 - largest.bead_diameter_mm) / 1.46) ** 2)
    assert largest.fillable_fraction < 0.75
    assert smaller.estimated_beads > largest.estimated_beads

    # Only the largest bead fits this budget. Tree generation has no later
    # filler pass, so its approximately 74.5% fit cannot satisfy a 75% target.
    result = autotune.auto_tune_bead_diameter(
        model, gen, contact, target_fill=0.75, steps=2,
        max_beads=largest.estimated_beads,
    )
    assert result.chosen.bead_diameter_mm == largest.bead_diameter_mm
    assert not result.met_target
    assert result.filler_diameters == []
    assert "더 작은 구슬" not in result.summary()


def test_tree_selection_does_not_depend_on_grid_filler_settings(tuning_inputs, geometry_spies):
    model, gen, contact = tuning_inputs
    geometry_spies["detect"].side_effect = lambda *args, **kwargs: [
        Polygon(), box(0, 0, 1.46, 1.46), Polygon()
    ]
    results = [autotune.auto_tune_bead_diameter(
        model, replace(gen, tree_enabled=True, fill_generations=generations),
        contact, target_fill=0.75,
    ) for generations in (0, 1, 3, 8)]
    assert len({result.chosen.bead_diameter_mm for result in results}) == 1
    assert all(result.met_target for result in results)
    assert all(result.chosen.fillable_fraction >= 0.75 for result in results)
    assert all(result.filler_diameters == [] for result in results)
