# -*- coding: utf-8 -*-
"""격자 배치가 실제로 최밀충전을 만드는지 검증."""

import math

import numpy as np
import pytest
from shapely.geometry import box

from pellet_support import (
    SupportBeadParams,
    bead_angle,
    support_bead_generate_centers,
)

ORIGIN = (0.0, 0.0)


def _params(**kw):
    base = dict(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08,
                edge_margin_ratio=0.0)
    base.update(kw)
    return SupportBeadParams(**base)


def _xy(centers):
    return np.array([[c.x, c.y] for c in centers])


def test_centers_stay_inside_the_region():
    region = box(0, 0, 10, 10)
    p = _params(edge_margin_ratio=0.5)
    centers = support_bead_generate_centers(region, p, 0, ORIGIN)
    assert len(centers) > 0
    margin = p.bead_diameter_mm * p.edge_margin_ratio
    safe = region.buffer(-margin)
    for c in centers:
        assert safe.distance(__import__("shapely").geometry.Point(c.x, c.y)) < 1e-9


def test_in_plane_neighbours_sit_at_pitch():
    """한 층 안에서 최근접 이웃 거리는 정확히 pitch 여야 한다."""
    p = _params()
    centers = support_bead_generate_centers(box(0, 0, 10, 10), p, 0, ORIGIN)
    pts = _xy(centers)
    # 가운데 점 하나를 골라 최근접 거리 확인 (경계 효과 회피)
    center = pts[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]
    d = np.linalg.norm(pts - center, axis=1)
    d = np.sort(d)[1:7]  # 자기 자신 제외, 육각 격자의 이웃 6개
    assert np.allclose(d, p.pitch_mm(), atol=1e-6)


def test_layer_shift_lands_on_the_hollow():
    """다음 층 구는 아래 세 구와 모두 pitch 만큼 떨어져야 한다."""
    p = _params(stagger_period=3)
    region = box(0, 0, 12, 12)
    a = _xy(support_bead_generate_centers(region, p, 0, ORIGIN))
    b = _xy(support_bead_generate_centers(region, p, 1, ORIGIN))

    layer_h = p.layer_height_mm()
    pitch = p.pitch_mm()

    # B 층 가운데 구를 하나 고른다
    probe = b[np.argmin(np.linalg.norm(b - b.mean(axis=0), axis=1))]
    # A 층 구들과의 3D 거리
    horizontal = np.linalg.norm(a - probe, axis=1)
    dist3d = np.sqrt(horizontal ** 2 + layer_h ** 2)
    touching = np.sort(dist3d)[:3]
    assert np.allclose(touching, pitch, atol=1e-6), touching


def test_abc_cycle_returns_to_a():
    """period 3 이면 3층 만에, period 2 면 2층 만에 원래 격자로 돌아온다."""
    region = box(0, 0, 8, 8)
    for period in (2, 3):
        p = _params(stagger_period=period)
        first = _xy(support_bead_generate_centers(region, p, 0, ORIGIN))
        same = _xy(support_bead_generate_centers(region, p, period, ORIGIN))
        assert np.allclose(np.sort(first, axis=0), np.sort(same, axis=0))


def test_intermediate_layers_are_actually_shifted():
    region = box(0, 0, 8, 8)
    p = _params(stagger_period=3)
    a = _xy(support_bead_generate_centers(region, p, 0, ORIGIN))
    b = _xy(support_bead_generate_centers(region, p, 1, ORIGIN))
    c = _xy(support_bead_generate_centers(region, p, 2, ORIGIN))
    assert not np.allclose(np.sort(a, axis=0)[:5], np.sort(b, axis=0)[:5])
    assert not np.allclose(np.sort(b, axis=0)[:5], np.sort(c, axis=0)[:5])


def test_straight_columns_do_not_shift():
    region = box(0, 0, 8, 8)
    p = _params(stagger_layers=False)
    a = _xy(support_bead_generate_centers(region, p, 0, ORIGIN))
    b = _xy(support_bead_generate_centers(region, p, 1, ORIGIN))
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_grid_is_stable_across_different_region_shapes():
    """영역 모양이 달라져도 격자점 좌표 자체는 같은 자리에 있어야 한다.

    원본 C++ 이 층별 bbox 를 원점으로 쓰던 버그를 막는 회귀 테스트.
    """
    p = _params()
    wide = support_bead_generate_centers(box(0, 0, 10, 10), p, 0, ORIGIN)
    narrow = support_bead_generate_centers(box(3, 3, 7, 7), p, 0, ORIGIN)

    wide_pts = {(round(c.x, 6), round(c.y, 6)) for c in wide}
    for c in narrow:
        assert (round(c.x, 6), round(c.y, 6)) in wide_pts


def test_thin_island_is_skipped_when_a_margin_is_required():
    """여백이 필요한데 조각이 그보다 얇으면 이 구슬로는 채우지 않는다.

    예전에는 원본 영역을 그대로 써서 구슬을 욱여넣었는데, 그러면 구슬이
    영역 밖으로 반지름만큼 삐져나가 모델을 파고들었다(실측 4.9% 충돌).
    이런 조각은 refine_thin_region 이 더 작은 구슬로 처리한다.
    """
    sliver = box(0, -0.15, 6, 0.15)          # 폭 0.3mm
    p = _params(edge_margin_ratio=0.5)        # 여백 0.5mm > 폭의 절반
    assert support_bead_generate_centers(sliver, p, 0, ORIGIN) == []


def test_thin_island_is_filled_when_no_margin_is_needed():
    """여백이 필요 없으면(모델과 충돌 위험이 없으면) 얇은 조각도 채운다."""
    sliver = box(0, -0.15, 6, 0.15)
    p = _params(edge_margin_ratio=0.5)
    centers = support_bead_generate_centers(
        sliver, p, 0, ORIGIN, edge_margin_override=0.0)
    assert len(centers) > 0


def test_snake_order_reverses_odd_rows_only():
    p = _params(snake_order=True)
    centers = support_bead_generate_centers(box(0, 0, 10, 10), p, 0, ORIGIN)
    rows = {}
    for c in centers:
        rows.setdefault(c.row, []).append(c.x)
    for row, xs in rows.items():
        if len(xs) < 2:
            continue
        if row % 2 == 0:
            assert xs == sorted(xs)
        else:
            assert xs == sorted(xs, reverse=True)


def test_bead_angle_cycles_with_period():
    p = _params(stagger_period=3)
    angles = [bead_angle(p, i) for i in range(4)]
    assert angles[0] == pytest.approx(0.0)
    assert angles[1] == pytest.approx(math.pi / 3)
    assert angles[2] == pytest.approx(2 * math.pi / 3)
    assert angles[3] == pytest.approx(0.0)


def test_empty_region_yields_no_centers():
    from shapely.geometry import Polygon

    assert support_bead_generate_centers(Polygon(), _params(), 0, ORIGIN) == []
