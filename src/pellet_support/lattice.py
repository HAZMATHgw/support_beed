# -*- coding: utf-8 -*-
"""bead 중심 좌표 배치. C++ ``support_bead_generate_centers`` 의 이식본.

원본과 달라진 곳은 두 군데뿐이다.

1. 격자 원점을 그 층 폴리곤의 bbox 가 아니라 오브젝트 고정 원점으로 받는다.
   기준 격자가 층마다 흔들리면 시프트를 정확히 줘도 아래층 hollow 에
   떨어지지 않는다.
2. 층 시프트를 ``% 2`` 대신 ``% stagger_period`` 로 돌려 ABC 순환을 만든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .params import SupportBeadParams
from .slicing import clean


@dataclass
class SupportBeadCenter:
    x: float
    y: float
    row: int = 0


def support_bead_generate_centers(
    region_geom,
    params: SupportBeadParams,
    support_layer_id: int,
    grid_origin: Tuple[float, float],
    pitch_override: Optional[float] = None,
) -> List[SupportBeadCenter]:
    """영역 안에 들어가는 격자점만 골라 bead 중심으로 반환."""
    centers: List[SupportBeadCenter] = []
    region_geom = clean(region_geom)
    if region_geom.is_empty:
        return centers

    # 채움 구슬은 '자기 지름' 이 아니라 '기본 격자' 위에 놓여야 하므로
    # 간격을 밖에서 지정할 수 있게 한다.
    pitch_mm = pitch_override if pitch_override else params.pitch_mm()
    if pitch_mm <= 0.0 or params.bead_diameter_mm <= 0.0:
        return centers

    edge_margin = params.bead_diameter_mm * params.edge_margin_ratio
    # 가장자리 여백은 조각(part)마다 따로 적용한다.
    #
    # 예전에는 지오메트리 전체를 한 번에 shrink 하고 "전부 사라졌을 때만"
    # 원본으로 되돌렸다. 그래서 넓은 조각과 얇은 조각이 섞여 있으면 —
    # 서포터 영역은 거의 항상 그렇다 — 얇은 조각만 조용히 증발해서 구슬이
    # 하나도 안 놓였다. 굵은 펠릿일수록 여백이 커져서 피해가 컸다.
    parts = region_geom.geoms if hasattr(region_geom, "geoms") else [region_geom]
    safe_parts = []
    for part in parts:
        if part.is_empty:
            continue
        shrunk = part.buffer(-edge_margin)
        # 여백을 주면 사라질 만큼 얇은 조각은 원본을 그대로 쓰고, 중심점이
        # 그 안에 드는지로만 판정한다.
        safe_parts.append(part if shrunk.is_empty else shrunk)
    if not safe_parts:
        return centers
    try:
        safe = clean(unary_union(safe_parts))
    except Exception:
        # flare 로 넓힌 폴리곤들이 서로 얽히면 GEOS 가 TopologyException 을
        # 던진다. 0 버퍼로 위상을 정리한 뒤 다시 시도하고, 그래도 안 되면
        # 조각을 하나씩 합쳐서 문제가 있는 조각만 버린다.
        repaired = []
        for part in safe_parts:
            try:
                fixed = part.buffer(0)
                if not fixed.is_empty:
                    repaired.append(fixed)
            except Exception:
                continue
        try:
            safe = clean(unary_union(repaired)) if repaired else Polygon()
        except Exception:
            merged = Polygon()
            for part in repaired:
                try:
                    merged = merged.union(part)
                except Exception:
                    continue
            safe = clean(merged)
    if safe.is_empty:
        return centers

    # 정삼각형 격자: 행 안에서는 pitch, 행과 행 사이는 pitch * sqrt(3)/2
    pitch_x = pitch_mm
    pitch_y = pitch_mm * math.sqrt(3.0) * 0.5

    # 최밀충전: 층 k 는 층 k-1 의 hollow 로 내려앉는다. 위를 향한 격자 삼각형의
    # hollow 는 그 무게중심 (pitch_x/2, pitch_y/3). 이 시프트를 반복하면
    # A -> B -> C 를 돌고 stagger_period 층 만에 A 로 돌아온다.
    layer_shift_x = layer_shift_y = 0.0
    if params.stagger_layers:
        period = max(2, params.stagger_period)
        k = support_layer_id % period
        layer_shift_x = k * (pitch_x / 2.0)
        layer_shift_y = k * (pitch_y / 3.0)

    # 행/열 인덱스는 그 층 안의 카운터가 아니라 오브젝트 전역 격자 인덱스다.
    # 그래야 행 패리티(= 반 칸 시프트)가 층끼리, 영역끼리 어긋나지 않는다.
    ox, oy = grid_origin
    base_x, base_y = ox + layer_shift_x, oy + layer_shift_y
    min_x, min_y, max_x, max_y = safe.bounds

    row_first = int(math.floor((min_y - base_y) / pitch_y))
    row_last = int(math.ceil((max_y - base_y) / pitch_y))

    for row in range(row_first, row_last + 1):
        y = base_y + row * pitch_y
        row_base_x = base_x + (0.0 if row % 2 == 0 else pitch_x / 2.0)
        col_first = int(math.floor((min_x - row_base_x) / pitch_x))
        col_last = int(math.ceil((max_x - row_base_x) / pitch_x))

        row_centers: List[SupportBeadCenter] = []
        for col in range(col_first, col_last + 1):
            x = row_base_x + col * pitch_x
            if safe.contains(Point(x, y)):
                row_centers.append(SupportBeadCenter(x, y, row))

        # Snake order: 짝수 행 좌->우, 홀수 행 우->좌. 이동 거리만 줄인다.
        if params.snake_order and (row % 2 != 0):
            row_centers.reverse()
        centers.extend(row_centers)
    return centers


def bead_angle(params: SupportBeadParams, support_layer_id: int) -> float:
    """짧은 압출 선분의 잔여 이방성을 층마다 돌려서 상쇄한다."""
    if not params.alternate_contact_angle:
        return 0.0
    period = max(2, params.stagger_period) if params.stagger_layers else 2
    return math.pi * (support_layer_id % period) / period
