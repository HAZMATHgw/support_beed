# -*- coding: utf-8 -*-
"""입력 검증 회귀 테스트.

여기 있는 케이스는 전부 예전에 **조용히 틀린 결과를 내거나 멈추던** 것들이다.
예를 들어 xy_clearance 를 음수로 주면 서포터가 모델을 파고든 채로 구슬이
오히려 더 많이 생성됐고, 초대형 모델은 메모리가 터질 때까지 그냥 멈췄다.
"""

import numpy as np
import pytest
import trimesh

from pellet_support import SupportGenParams, generate_support, make_params
from pellet_support.validation import (
    InvalidModelError,
    InvalidParameterError,
    TooManyBeadsError,
    validate_mesh,
)


@pytest.fixture(scope="module")
def model():
    leg = trimesh.creation.box(extents=[20, 20, 10])
    leg.apply_translation([0, 0, 5])
    top = trimesh.creation.box(extents=[40, 40, 4])
    top.apply_translation([0, 0, 12])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def _run(model, make_kw=None, gen_kw=None, detail=0):
    contact, body = make_params(**(make_kw or {"nozzle_diameter_mm": 1.0}))
    gen = SupportGenParams(
        **{**dict(layer_height_mm=contact.layer_height_mm()), **(gen_kw or {})}
    )
    return generate_support(model, gen, contact, body, detail=detail, verbose=False)


# --- 구슬 파라미터 -------------------------------------------------------

@pytest.mark.parametrize("overlap", [1.0, 1.5, 2.0])
def test_overlap_at_or_above_one_is_rejected(overlap):
    """겹침이 1 이상이면 pitch <= 0 이라 격자가 성립하지 않는다."""
    with pytest.raises(InvalidParameterError, match="겹침"):
        make_params(nozzle_diameter_mm=1.0, overlap=overlap)


def test_negative_overlap_is_rejected():
    """음수 겹침은 구슬이 서로 닿지 않아 서포터가 무너진다."""
    with pytest.raises(InvalidParameterError, match="겹침"):
        make_params(nozzle_diameter_mm=1.0, overlap=-0.5)


@pytest.mark.parametrize("nozzle", [0.0, -1.0])
def test_nonpositive_nozzle_is_rejected(nozzle):
    with pytest.raises(InvalidParameterError, match="노즐"):
        make_params(nozzle_diameter_mm=nozzle)


@pytest.mark.parametrize("bead", [0.0, -1.0])
def test_nonpositive_bead_diameter_is_rejected(bead):
    """0 은 falsy 라서 예전에는 '지정 안 함'으로 조용히 바뀌었다."""
    with pytest.raises(InvalidParameterError, match="구슬 지름"):
        make_params(nozzle_diameter_mm=1.0, bead_diameter_mm=bead)


def test_bead_diameter_none_still_uses_auto_value():
    """None 일 때만 자동값(노즐의 절반)을 쓴다."""
    contact, _ = make_params(nozzle_diameter_mm=8.0, bead_diameter_mm=None)
    assert contact.bead_diameter_mm == pytest.approx(4.0)


@pytest.mark.parametrize("ratio", [0.0, -1.0, 1.5, 5.0])
def test_body_bead_ratio_out_of_range_is_rejected(ratio):
    """1 을 넘으면 몸통 구슬이 인터페이스보다 커져 공유 격자가 깨진다."""
    with pytest.raises(InvalidParameterError, match="몸통 구슬 비율"):
        make_params(nozzle_diameter_mm=1.0, body_bead_ratio=ratio)


# --- 영역 파라미터 -------------------------------------------------------

@pytest.mark.parametrize("angle", [0, 90, -30, 200])
def test_overhang_angle_out_of_range_is_rejected(model, angle):
    with pytest.raises(InvalidParameterError, match="오버행 각도"):
        _run(model, gen_kw={"overhang_angle_deg": angle})


def test_negative_xy_clearance_is_rejected(model):
    """음수면 서포터가 모델을 파고드는데, 예전에는 구슬이 더 많이 생성됐다."""
    with pytest.raises(InvalidParameterError, match="XY 여유"):
        _run(model, gen_kw={"xy_clearance_mm": -5})


@pytest.mark.parametrize("layers", [0, -3])
def test_contact_layers_below_one_is_rejected(model, layers):
    with pytest.raises(InvalidParameterError, match="인터페이스 층"):
        _run(model, gen_kw={"contact_layers": layers})


def test_negative_z_gap_is_rejected(model):
    with pytest.raises(InvalidParameterError, match="Z 간격"):
        _run(model, gen_kw={"contact_z_gap_layers": -5})


@pytest.mark.parametrize("h", [0, -1])
def test_nonpositive_layer_height_is_rejected(model, h):
    with pytest.raises(InvalidParameterError, match="층높이"):
        _run(model, gen_kw={"layer_height_mm": h})


@pytest.mark.parametrize("h", [0, -1])
def test_nonpositive_detection_height_is_rejected(model, h):
    with pytest.raises(InvalidParameterError, match="탐지 두께"):
        _run(model, gen_kw={"detection_layer_height_mm": h})


@pytest.mark.parametrize("detail", [-1, 4, 99])
def test_sphere_detail_out_of_range_is_rejected(model, detail):
    with pytest.raises(InvalidParameterError, match="세분화"):
        _run(model, detail=detail)


# --- 모델 -----------------------------------------------------------------

def test_empty_mesh_is_rejected():
    with pytest.raises(InvalidModelError, match="삼각형"):
        validate_mesh(trimesh.Trimesh())


def test_mesh_with_vertices_but_no_faces_is_rejected():
    mesh = trimesh.Trimesh(
        vertices=np.random.rand(10, 3), faces=np.zeros((0, 3), dtype=int)
    )
    with pytest.raises(InvalidModelError, match="삼각형"):
        validate_mesh(mesh)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_vertices_are_rejected(bad):
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.vertices[0] = [bad] * 3
    with pytest.raises(InvalidModelError, match="NaN"):
        validate_mesh(mesh)


def test_zero_height_model_is_rejected():
    with pytest.raises(InvalidModelError, match="높이"):
        validate_mesh(trimesh.creation.box(extents=[20, 20, 0]))


# --- 메모리 폭발 방지 ------------------------------------------------------

def test_huge_model_is_stopped_before_running_out_of_memory():
    """예전에는 그냥 멈춰서 서버가 죽었다. 미리 막고 대안을 알려준다."""
    base = trimesh.creation.box(extents=[2000, 2000, 500])
    top = trimesh.creation.box(extents=[3000, 3000, 50])
    top.apply_translation([0, 0, 300])
    mesh = trimesh.util.concatenate([base, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    with pytest.raises(TooManyBeadsError) as exc:
        _run(mesh)
    assert "bead-diameter" in str(exc.value)  # 무엇을 바꿔야 하는지 알려줘야 한다


def test_max_beads_can_be_raised(model):
    """상한은 사용자가 올릴 수 있어야 한다."""
    result = _run(model, gen_kw={"max_beads": 10_000_000})
    assert result.plan is not None


# --- 정상 입력은 그대로 통과해야 한다 --------------------------------------

def test_valid_input_still_works(model):
    result = _run(model)
    assert result.plan is not None
    assert sum(len(l["beads"]) for l in result.plan.layers) > 100


def test_warns_when_bead_is_too_big_for_the_support_regions():
    """구슬이 영역보다 크면 아무리 잘 깔아도 끊긴다. 미리 알려줘야 한다.

    노즐 5mm(구슬 2.5mm)로 31mm 크기 모델을 받치면 서포터 영역의 절반이
    구슬보다 좁아서 채울 수가 없었다.
    """
    from pellet_support.validation import check_bead_fits_regions, fillable_fraction
    from shapely.geometry import box as shapely_box

    # 폭 1mm 짜리 가느다란 영역들
    thin = [shapely_box(0, 0, 50, 1.0) for _ in range(5)]
    contact, _ = make_params(nozzle_diameter_mm=5.0)   # 구슬 2.5mm

    assert fillable_fraction(thin, contact.bead_diameter_mm) == pytest.approx(0.0)
    msg = check_bead_fits_regions(thin, contact, 5.0)
    assert msg is not None
    assert "bead-diameter" in msg


def test_no_warning_when_bead_fits_comfortably():
    from pellet_support.validation import check_bead_fits_regions
    from shapely.geometry import box as shapely_box

    wide = [shapely_box(0, 0, 50, 50)]
    contact, _ = make_params(nozzle_diameter_mm=1.0)
    assert check_bead_fits_regions(wide, contact, 1.0) is None
