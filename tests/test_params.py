# -*- coding: utf-8 -*-
"""충전 기하 계산이 실제 3D 기하와 일치하는지 검증."""

import math
from dataclasses import replace

import numpy as np
import pytest

from pellet_support import (
    TETRA_HEIGHT,
    SupportBeadParams,
    support_bead_body_params,
    support_bead_contact_params,
    unify_lattice,
)


def test_tetra_height_matches_actual_geometry():
    """세 구의 hollow 에 앉은 구의 높이를 좌표로 직접 풀어 상수와 대조."""
    pitch = 1.0
    p1 = np.array([0.0, 0.0])
    p2 = np.array([pitch, 0.0])
    p3 = np.array([pitch / 2, pitch * math.sqrt(3) / 2])
    centroid = (p1 + p2 + p3) / 3

    # hollow 는 정삼각형의 무게중심이고, 코드가 쓰는 (pitch_x/2, pitch_y/3) 와 같다
    pitch_y = pitch * math.sqrt(3) / 2
    assert centroid[0] == pytest.approx(pitch / 2)
    assert centroid[1] == pytest.approx(pitch_y / 3)

    # 세 구 모두와 pitch 만큼 떨어지려면 z 가 얼마여야 하는가
    horizontal = np.linalg.norm(centroid - p1)
    z = math.sqrt(pitch ** 2 - horizontal ** 2)
    assert z == pytest.approx(TETRA_HEIGHT)


def test_pitch_and_layer_height():
    p = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08)
    assert p.pitch_mm() == pytest.approx(0.92)
    assert p.layer_height_mm() == pytest.approx(0.92 * TETRA_HEIGHT)


def test_straight_columns_use_pitch_as_layer_height():
    p = SupportBeadParams(lattice_overlap_ratio=0.08, stagger_layers=False)
    assert p.layer_height_mm() == pytest.approx(p.pitch_mm())
    assert p.coordination_number() == 8


def test_tangent_spheres_have_no_contact_disc():
    p = SupportBeadParams(lattice_overlap_ratio=0.0)
    assert p.contact_disc_radius_mm() == pytest.approx(0.0)
    assert p.coordination_number() == 12


def test_contact_disc_grows_with_sqrt_of_overlap():
    """접촉원은 sqrt(delta) 로, 잃는 부피는 delta 로 늘어난다."""
    d = 1.0
    small = SupportBeadParams(bead_diameter_mm=d, lattice_overlap_ratio=0.02)
    large = SupportBeadParams(bead_diameter_mm=d, lattice_overlap_ratio=0.08)

    # delta 가 4배면 접촉원 반지름은 약 2배
    ratio = large.contact_disc_radius_mm() / small.contact_disc_radius_mm()
    assert ratio == pytest.approx(2.0, rel=0.02)

    # 해석해와 직접 대조
    for p in (small, large):
        r, pitch = 0.5 * d, p.pitch_mm()
        assert p.contact_disc_radius_mm() == pytest.approx(
            math.sqrt(r * r - 0.25 * pitch * pitch)
        )


def test_packing_fraction_in_expected_band():
    """맞닿기만 하면 최밀충전 이론값 0.74, 누를수록 올라간다."""
    tangent = SupportBeadParams(lattice_overlap_ratio=0.0)
    assert tangent.packing_fraction() == pytest.approx(0.7405, abs=0.002)

    pressed = SupportBeadParams(lattice_overlap_ratio=0.08)
    assert 0.85 < pressed.packing_fraction() < 0.95
    assert pressed.packing_fraction() > tangent.packing_fraction()


def test_mm3_per_mm_carries_the_bead_volume():
    """선분 길이를 곱하면 목표 구 부피가 그대로 나와야 한다."""
    p = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.06)
    seg_len = p.bead_diameter_mm * p.segment_ratio
    assert p.mm3_per_mm() * seg_len == pytest.approx(p.bead_volume_mm3())


def test_bead_volume_is_less_than_full_sphere():
    p = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08)
    full = 4.0 / 3.0 * math.pi * 0.5 ** 3
    assert 0 < p.bead_volume_mm3() < full


def test_unify_lattice_forces_a_single_pitch():
    """body 는 구만 작아지고 pitch 는 contact 와 같아야 한다."""
    contact = support_bead_contact_params(1.0)
    body = support_bead_body_params(1.0, bead_ratio=0.97)
    contact, body = unify_lattice(contact, body)

    assert body.pitch_mm() == pytest.approx(contact.pitch_mm())
    assert body.bead_diameter_mm < contact.bead_diameter_mm
    # 같은 격자 위에서 구가 작으니 겹침도 더 작다 = 더 잘 부서진다
    assert body.lattice_overlap_ratio < contact.lattice_overlap_ratio
    assert body.layer_height_mm() == pytest.approx(contact.layer_height_mm())


def test_bead_diameter_defaults_to_half_the_nozzle():
    """노즐 지름을 그대로 구 지름으로 쓰지 않는다 — 표준 FDM 이 층높이를
    노즐 지름의 25~75% 로 쓰는 것과 같은 논리. 기본값은 절반."""
    contact = support_bead_contact_params(nozzle_diameter_mm=8.0)
    assert contact.bead_diameter_mm == pytest.approx(4.0)


def test_bead_diameter_can_be_set_independently_of_nozzle():
    """노즐이 굵어도 구슬은 사용자가 원하는 만큼 작게 잡을 수 있다."""
    contact = support_bead_contact_params(nozzle_diameter_mm=8.0, bead_diameter_mm=1.0)
    assert contact.bead_diameter_mm == pytest.approx(1.0)


def test_smaller_bead_shrinks_the_dead_zone_at_large_nozzle():
    """노즐은 그대로 두고 구슬만 줄이면 gap+interface 사각지대가 준다."""
    big_bead = support_bead_contact_params(nozzle_diameter_mm=8.0)          # 4.0mm
    small_bead = support_bead_contact_params(nozzle_diameter_mm=8.0,
                                             bead_diameter_mm=1.0)          # 1.0mm

    dead_big = 3 * big_bead.layer_height_mm()      # gap(1) + interface(2) 층
    dead_small = 3 * small_bead.layer_height_mm()
    assert dead_small < dead_big
    assert dead_small == pytest.approx(dead_big * (1.0 / 4.0), rel=0.01)


def test_anisotropic_packing_is_opt_in():
    """가로/세로를 지정하지 않으면 기존 등방 기하 그대로여야 한다."""
    iso = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08)
    assert iso.lateral_overlap() == pytest.approx(0.08)
    assert iso.vertical_overlap() == pytest.approx(0.08)
    assert iso.layer_height_mm() == pytest.approx(iso.pitch_mm() * TETRA_HEIGHT)
    assert iso.lateral_neck_area_mm2() == pytest.approx(iso.vertical_neck_area_mm2())


def test_anisotropic_strengthens_lateral_and_weakens_vertical():
    """좌우 흔들림에 강해지고, 세로는 오히려 약해져 분리가 쉬워진다."""
    iso = SupportBeadParams(bead_diameter_mm=0.5, lattice_overlap_ratio=0.08)
    ani = SupportBeadParams(bead_diameter_mm=0.5,
                            lateral_overlap_ratio=0.14,
                            vertical_overlap_ratio=0.04)
    assert ani.lateral_neck_area_mm2() > iso.lateral_neck_area_mm2()
    assert ani.vertical_neck_area_mm2() < iso.vertical_neck_area_mm2()


def test_anisotropic_layer_height_keeps_vertical_distance():
    """층 높이는 아래 세 구슬과의 거리가 설계값이 되도록 역산돼야 한다."""
    import math

    p = SupportBeadParams(bead_diameter_mm=0.5,
                          lateral_overlap_ratio=0.14,
                          vertical_overlap_ratio=0.04)
    pitch, h = p.pitch_mm(), p.layer_height_mm()
    # 아래층 세 구슬의 무게중심까지 수평거리 = pitch/sqrt(3)
    actual = math.sqrt((pitch / math.sqrt(3)) ** 2 + h ** 2)
    assert actual == pytest.approx(p.vertical_neighbor_distance_mm(), rel=1e-9)


def test_unify_lattice_shares_pitch_under_anisotropy():
    """몸통 구슬이 작아도 같은 격자 위에 놓여야 한다."""
    contact = support_bead_contact_params(1.0)
    body = support_bead_body_params(1.0, bead_ratio=0.97)
    contact = replace(contact, lateral_overlap_ratio=0.14,
                      vertical_overlap_ratio=0.04)
    body = replace(body, lateral_overlap_ratio=0.14,
                   vertical_overlap_ratio=0.04)
    contact, body = unify_lattice(contact, body)
    assert body.pitch_mm() == pytest.approx(contact.pitch_mm())
    assert body.vertical_neighbor_distance_mm() == pytest.approx(
        contact.vertical_neighbor_distance_mm())
    assert body.layer_height_mm() == pytest.approx(contact.layer_height_mm())


def test_breakaway_shows_shake_proof_but_mallet_breakable():
    """흔들림에는 안 부서지고 망치에는 부서지는 설계인지 확인.

    구슬 질량은 지름의 세제곱, 넥 면적은 제곱으로 줄어서 작은 구슬일수록
    관성력 대비 넥이 압도적으로 강해진다. 반대로 넥의 절대 강도는 1N 안팎이라
    망치 타격(100~500N)에는 쉽게 끊어진다.
    """
    from pellet_support.simulate import breakaway_analysis

    p = SupportBeadParams(bead_diameter_mm=0.5,
                          lateral_overlap_ratio=0.14,
                          vertical_overlap_ratio=0.04)
    res = breakaway_analysis(p)

    assert res.shake_margin > 1000        # 흔들림으로는 절대 안 부서짐
    assert res.force_per_neck_n < 10      # 망치로는 쉽게 부서짐
    assert res.model_wall_ratio > 10      # 모델보다 훨씬 약해서 서포터만 부서짐


def test_weaker_vertical_overlap_makes_it_easier_to_break():
    from pellet_support.simulate import breakaway_analysis

    strong = SupportBeadParams(bead_diameter_mm=0.5, lattice_overlap_ratio=0.08)
    weak = SupportBeadParams(bead_diameter_mm=0.5,
                             lateral_overlap_ratio=0.16,
                             vertical_overlap_ratio=0.02)
    assert (breakaway_analysis(weak).force_per_neck_n
            < breakaway_analysis(strong).force_per_neck_n)
