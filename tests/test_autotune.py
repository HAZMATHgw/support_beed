# -*- coding: utf-8 -*-
"""구슬 크기 자동 선택 검증."""

import pytest
import trimesh

from pellet_support import SupportGenParams, make_params
from pellet_support.autotune import auto_tune_bead_diameter


@pytest.fixture(scope="module")
def model():
    leg = trimesh.creation.box(extents=[12, 12, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[36, 36, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def _tune(model, nozzle, **kw):
    contact, _ = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    return auto_tune_bead_diameter(model, gen, contact, **kw)


def test_chosen_bead_is_within_printable_range(model):
    """고른 구슬은 인쇄 가능 최소보다 크고 노즐 절반보다 작아야 한다."""
    nozzle = 5.0
    res = _tune(model, nozzle)
    assert nozzle * 0.35 - 1e-9 <= res.chosen.bead_diameter_mm <= nozzle * 0.5 + 1e-9


def test_picks_largest_bead_that_meets_the_target(model):
    """목표를 만족하는 것 중 가장 큰 구슬을 골라야 한다.

    필요 이상으로 작게 잡으면 구슬 수가 세제곱으로 늘어 인쇄만 오래 걸린다.
    """
    res = _tune(model, 1.0, target_fill=0.5)
    meeting = [c for c in res.candidates if c.fillable_fraction >= 0.5]
    assert meeting, "목표를 만족하는 후보가 있어야 하는 조건"
    assert res.met_target
    assert res.chosen.bead_diameter_mm == max(
        c.bead_diameter_mm for c in meeting)


def test_reports_when_target_cannot_be_met(model):
    """달성 불가능한 목표를 주면 조용히 넘어가지 말고 알려야 한다."""
    res = _tune(model, 5.0, target_fill=0.999)
    assert not res.met_target
    assert "노즐" in res.summary()


def test_smaller_bead_always_fills_more(model):
    """구슬이 작을수록 채울 수 있는 영역 비율이 커져야 한다(단조성)."""
    res = _tune(model, 5.0)
    ordered = sorted(res.candidates, key=lambda c: c.bead_diameter_mm)
    fills = [c.fillable_fraction for c in ordered]
    assert fills == sorted(fills, reverse=True), fills


def test_max_beads_limit_is_respected(model):
    """구슬 수 상한을 주면 그걸 넘는 후보는 고르지 않는다."""
    res = _tune(model, 1.0, target_fill=0.0, max_beads=10_000_000)
    assert res.chosen.estimated_beads <= 10_000_000
    assert res.met_target


def test_impossible_bead_limit_picks_the_fewest_beads(model):
    """상한을 만족하는 후보가 하나도 없으면 개수가 가장 적은 것을 고른다.

    품질만 좇아 더 작은 구슬을 고르면 개수가 오히려 더 늘어난다.
    """
    res = _tune(model, 1.0, target_fill=0.0, max_beads=1)
    assert not res.met_target
    assert res.chosen.estimated_beads == min(
        c.estimated_beads for c in res.candidates)


def test_filler_sizes_are_reported(model):
    """세분 구슬로 메울 수 있는지, 크기가 얼마인지 알려줘야 한다."""
    res = _tune(model, 1.0)
    assert res.filler_diameters, "세분 구슬 크기가 비어 있다"
    # 세대마다 작아지고, 인쇄 가능 최소보다는 커야 한다
    assert res.filler_diameters == sorted(res.filler_diameters, reverse=True)
    assert min(res.filler_diameters) >= 1.0 * 0.35 - 1e-6


def test_warns_when_no_room_for_filler_beads(model):
    """기본 구슬이 이미 최소 크기면 틈을 메울 수 없다는 걸 알려야 한다."""
    contact, _ = make_params(nozzle_diameter_mm=5.0)
    gen = SupportGenParams(nozzle_diameter_mm=5.0,
                           layer_height_mm=contact.layer_height_mm())
    res = auto_tune_bead_diameter(model, gen, contact)
    if not res.filler_diameters:
        assert "여유가 없" in res.summary()
