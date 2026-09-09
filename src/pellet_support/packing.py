# -*- coding: utf-8 -*-
"""다중 크기 구슬 충전.

기본 격자 하나로만 채우면 두 가지 빈 공간이 남는다.

1. **좁은 영역** — 영역 폭이 구슬보다 좁으면 격자점이 하나도 안 들어가서
   구슬이 아예 안 놓인다. 서포터 영역은 바닥까지 이어져 있는데 구슬 사슬만
   끊기는 원인이 바로 이것이었다(배 모델에서 비드의 55%가 여기서 떴다).
2. **격자 사이 빈틈** — 최밀충전이라도 구슬 사이에는 팔면체·사면체 모양의
   빈틈이 남는다. 맞닿기만 한 경우 충전율은 0.74 가 상한이다.

여기서는 둘 다 더 작은 구슬로 메운다. 3D 프린터가 층을 쌓듯, 큰 구슬로 먼저
채우고 남은 틈에 작은 구슬을 넣는 방식이다.

빈틈의 크기는 격자에서 정확히 정해진다(pitch 를 p, 기본 구슬 반지름을 R 이라 할 때):

- 팔면체 빈틈: 주변 6개 구슬 중심에서 ``0.70711*p`` -> 지름 ``2*(0.70711p - R)``
- 사면체 빈틈: 주변 4개 구슬 중심에서 ``0.61237*p`` -> 지름 ``2*(0.61237p - R)``

**물리적 한계**: 채움 구슬은 노즐이 뽑을 수 있는 최소 크기보다 커야 한다.
겹침 0.08 기준 팔면체 빈틈은 기본 구슬의 0.30배, 사면체는 0.13배라서,
기본 구슬이 충분히 크지 않으면 채움 자체가 불가능하다. 이 모듈은 인쇄 불가능한
크기의 채움 구슬은 만들지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .lattice import support_bead_generate_centers
from .params import SupportBeadParams, SupportBeadRegion
from .slicing import clean

#: 팔면체 빈틈 중심까지의 거리 / pitch
OCTA_SITE_RATIO = 0.7071067811865476     # 1/sqrt(2)
#: 사면체 빈틈 중심까지의 거리 / pitch
TETRA_SITE_RATIO = 0.6123724356957945    # sqrt(3/8)
#: 사면체 빈틈의 높이 위치 / pitch (아래 층 평면 기준)
TETRA_Z_LOW = 0.2041241452319315         # sqrt(1/24)
TETRA_Z_HIGH = TETRA_SITE_RATIO


def octahedral_bead_diameter(pitch_mm: float, base_radius_mm: float) -> float:
    """팔면체 빈틈에 들어갈 수 있는 최대 구슬 지름."""
    return 2.0 * (OCTA_SITE_RATIO * pitch_mm - base_radius_mm)


def tetrahedral_bead_diameter(pitch_mm: float, base_radius_mm: float) -> float:
    """사면체 빈틈에 들어갈 수 있는 최대 구슬 지름."""
    return 2.0 * (TETRA_SITE_RATIO * pitch_mm - base_radius_mm)


def refine_thin_region(
    region,
    params: SupportBeadParams,
    support_layer_id: int,
    grid_origin: Tuple[float, float],
    min_bead_mm: float,
    max_steps: int = 6,
    shrink: float = 0.7,
) -> Tuple[List, float]:
    """구슬이 하나도 안 들어간 좁은 영역을 더 작은 구슬로 다시 시도한다.

    지름을 절반씩 줄여 가며 격자점이 들어갈 때까지 재시도하고, 인쇄 가능한
    최소 크기 아래로는 내려가지 않는다. (센터, 지름) 을 돌려준다.
    """
    region = clean(region)
    if region.is_empty:
        return [], params.bead_diameter_mm

    diameter = params.bead_diameter_mm
    for _ in range(max_steps):
        diameter *= shrink
        if diameter < min_bead_mm:
            break
        smaller = replace(params, bead_diameter_mm=diameter)
        centers = support_bead_generate_centers(
            region, smaller, support_layer_id, grid_origin
        )
        if centers:
            return centers, diameter
    return [], diameter


def interstitial_beads(
    layer_centers_by_shift,
    base_params: SupportBeadParams,
    min_bead_mm: float,
    include_tetrahedral: bool = True,
) -> List[dict]:
    """격자 사이 빈틈(팔면체·사면체)에 넣을 채움 구슬을 만든다.

    ``layer_centers_by_shift`` 는 ``(z_center, shift_index, [(x, y), ...])`` 의
    목록으로, 각 구슬 층의 격자 오프셋과 좌표를 담는다.

    인쇄 불가능한 크기의 채움 구슬은 만들지 않는다.
    """
    pitch = base_params.pitch_mm()
    r_base = 0.5 * base_params.bead_diameter_mm
    out: List[dict] = []

    d_oct = octahedral_bead_diameter(pitch, r_base)
    d_tet = tetrahedral_bead_diameter(pitch, r_base)

    layers = sorted(layer_centers_by_shift, key=lambda t: t[0])
    for idx in range(len(layers) - 1):
        z_lo, _shift_lo, pts_lo = layers[idx]
        z_hi, _shift_hi, pts_hi = layers[idx + 1]
        if not pts_lo or not pts_hi:
            continue

        # 사면체 빈틈: 위층 구슬 XY 아래쪽 / 아래층 구슬 XY 위쪽
        if include_tetrahedral and d_tet >= min_bead_mm:
            for (x, y) in pts_hi:
                out.append({"x": x, "y": y, "z": z_lo + TETRA_Z_LOW * pitch,
                            "d": d_tet, "kind": "tetra"})
            for (x, y) in pts_lo:
                out.append({"x": x, "y": y, "z": z_lo + TETRA_Z_HIGH * pitch,
                            "d": d_tet, "kind": "tetra"})
    return out, d_oct, d_tet
