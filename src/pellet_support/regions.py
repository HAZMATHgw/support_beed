# -*- coding: utf-8 -*-
"""오버행을 찾아 서포터가 차지할 영역을 층별로 계산한다.

슬라이서 내부에서는 이미 계산되어 넘어오는 부분이지만, 이 도구는 슬라이서
없이 돌아가야 하므로 직접 구한다.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .params import SupportGenParams
from .slicing import clean, drop_small


def _fill_holes(geom):
    """단면의 내부 구멍을 메워 바깥 윤곽만 남긴다.

    각 폴리곤을 그 exterior 링으로 바꾸므로, 닫힌 공동(구멍)만 메워지고
    오목한 만입부(U자 홈, 아치 아래)는 그대로 남는다.
    """
    geom = clean(geom)
    if geom.is_empty:
        return geom
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    filled = [Polygon(p.exterior) for p in parts if hasattr(p, "exterior")]
    return unary_union(filled) if filled else geom


def build_support_regions(
    slices: Sequence, params: SupportGenParams, detection_h: float = None
) -> Tuple[List, List]:
    """(support, contact) 를 층별 폴리곤 리스트로 반환.

    ``slices`` 는 **탐지용 얇은 층**으로 자른 단면이다. 구슬 격자 간격이 아니라
    모델 형상을 제대로 볼 수 있는 해상도여야 한다. 탐지 해상도를 구슬 크기에
    묶어 두면, 굵은 펠릿을 쓸 때 층이 듬성듬성해져서 그 사이의 오버행을 통째로
    놓치고 "서포터가 필요 없다" 는 잘못된 결론이 나온다.

    - support[i] : i 층에서 서포터가 차지하는 XY 영역
    - contact[i] : 그중 모델 아랫면 바로 밑인 영역
    """
    n = len(slices)
    det_h = detection_h or params.layer_height_mm
    ang = math.radians(max(1.0, min(89.0, params.overhang_angle_deg)))
    # 한 층 올라갈 때 자립 가능한 수평 이동량
    step = det_h / math.tan(ang)

    # 사용자가 지정한 값은 '구슬 층' 기준이므로 mm 로 바꾼 뒤 탐지 층 수로 환산한다.
    # 그래야 탐지 해상도를 바꿔도 실제 틈과 인터페이스 두께가 그대로 유지된다.
    gap = max(0, int(round(
        params.contact_z_gap_layers * params.layer_height_mm / det_h)))
    cl = max(1, int(round(
        params.contact_layers * params.layer_height_mm / det_h)))

    # (1) 아래가 비어 있는 영역 = 오버행
    overhang: List = [Polygon()] * n
    for i in range(1, n):
        cur, below = clean(slices[i]), clean(slices[i - 1])
        if cur.is_empty:
            continue
        grown = below.buffer(step) if not below.is_empty else Polygon()
        oh = cur.difference(grown) if not grown.is_empty else cur
        overhang[i] = drop_small(oh, params.min_island_area_mm2)

    # (2) 위에서 아래로 누적하며 기둥을 만든다
    support: List = [Polygon()] * n
    acc = Polygon()
    # 아래로 내려갈수록 조금씩 넓힌다(flare).
    #
    # 이유가 두 가지다. 첫째, 플레이트가 좌우로 흔들릴 때 밑이 좁은 기둥은
    # 그대로 넘어간다 — 바닥을 넓혀야 전도에 버틴다. 둘째, 위에서 따로 시작한
    # 기둥들이 내려오면서 서로 만나 하나로 합쳐진다. 예전에는 배 모델에서
    # 서포터가 169개 조각으로 파편화됐다.
    flare = max(0.0, params.flare_mm_per_layer)
    for i in range(n - 1, -1, -1):
        src = i + 1 + gap  # gap 층만큼 띄운 뒤부터 지지 시작
        if src < n:
            acc = clean(unary_union([acc, overhang[src]]))
        if acc.is_empty:
            continue
        if flare > 0:
            try:
                acc = clean(acc.buffer(flare))
            except Exception:
                # 넓히다가 폴리곤이 자기 교차하면 GEOS 가 예외를 던진다.
                # 위상을 정리한 뒤 한 번 더 시도하고, 그래도 안 되면 이번
                # 층은 넓히지 않고 넘어간다(서포터가 사라지는 것보다 낫다).
                try:
                    acc = clean(clean(acc).buffer(0).buffer(flare))
                except Exception:
                    pass
        model = clean(slices[i])
        if not model.is_empty and not params.allow_internal_supports:
            # 단면의 '구멍'을 메운 바깥 윤곽만 남긴다. 그 안쪽은 사방이 모델로
            # 둘러싸인 닫힌 공동이라, 거기에 구슬을 채워봐야 출력 후 꺼낼 수가
            # 없다. 속이 빈 상자를 넣으면 서포터의 100%가 여기에 갇혔었다.
            #
            # 바깥 윤곽만 채우므로, 아치 아래나 U자 홈처럼 위가 트인 공간은
            # 그대로 남는다(그런 곳은 폴리곤 외곽선 바깥이라 영향받지 않는다).
            model = _fill_holes(model)
        blocked = (
            model.buffer(params.xy_clearance_mm) if not model.is_empty else Polygon()
        )
        sup = acc.difference(blocked) if not blocked.is_empty else acc
        support[i] = drop_small(sup, params.min_island_area_mm2 * 0.25)
        # acc 자체는 깎지 않는다.
        #
        # 완만한 경사면의 오버행 링은 폭이 0.3mm 남짓인데 xy_clearance(0.8mm)로
        # 깎으면 통째로 사라진다. 예전에는 여기서 acc = sup 으로 되먹임해서
        # 그 순간 기둥이 죽어 버렸고, 결과적으로 서포터가 바닥까지 못 내려가고
        # 모델 표면에 껍질처럼 몇 층만 붙어 있었다.
        #
        # acc 를 그대로 두면, 지금 층에서는 모델에 막혀 잘린 부분이 모델이
        # 좁아지는 아래 층에서 되살아나 제대로 된 기둥이 된다. 끝까지 막혀
        # 있으면 그 위쪽은 자연히 모델 윗면에 얹히게 된다.

    # (2c) 접지 검사: 아래에 아무것도 없는 서포터는 지운다.
    #
    # 위 루프는 acc 를 깎지 않고 내려가므로, 아래층이 모델에 완전히 막히면
    # 그 층의 support 만 비고 위층은 그대로 남는다. 그 결과 기둥 전체가
    # 공중에 뜬다 — 배 모델에서 전체 구슬의 56%(111,780개)가 z=10.77 부터
    # 베드도 모델도 없이 떠 있었다.
    #
    # 아래에서 위로 훑으면서, 바로 아래 층의 '살아남은 서포터' 또는 '모델'과
    # 닿지 않는 조각을 버린다. 한 층에서 버려지면 그 위도 연쇄적으로 버려진다.
    # 톨러런스는 수치 오차만 흡수할 만큼 작아야 한다. 여유를 크게 주면
    # 선실 안에 떠 있는 서포터가 '옆에 있는 선체 벽'에 닿았다고 통과해 버린다.
    # 실제로 바로 아래에서 받쳐 주는 것과 XY 가 겹쳐야만 접지로 인정한다.
    tol = det_h * 0.5
    ground = None  # 첫 층은 베드에 얹히므로 검사 없이 통과
    for i in range(n):
        cur = clean(support[i])
        if cur.is_empty:
            ground = Polygon()
            continue
        if i == 0:
            support[i] = cur
            ground = cur
            continue
        # 아래 층에서 이 조각을 받쳐 줄 수 있는 것: 살아남은 서포터 + 모델 실물
        below_model = clean(slices[i - 1])
        base_parts = [g for g in (ground, below_model) if g is not None and not g.is_empty]
        if not base_parts:
            support[i] = Polygon()
            ground = Polygon()
            continue
        base = clean(unary_union(base_parts))
        if tol > 0:
            base = base.buffer(tol)
        parts = cur.geoms if hasattr(cur, "geoms") else [cur]
        keep = [p for p in parts if not p.is_empty and p.intersection(base).area > 0]
        support[i] = clean(unary_union(keep)) if keep else Polygon()
        ground = support[i]

    # (2d) 인터페이스 재계산은 접지 검사 뒤에 해야 한다(아래 (3) 참고).


    #
    # 흔들림에 넘어지느냐는 바닥 폭이 결정한다. flare 로 전체를 넓히면
    # 위쪽까지 같이 두꺼워져 구슬 수가 폭증하는데(0.08->0.35 로 올리면 8배),
    # 브림은 바닥 근처에만 구슬을 더하므로 훨씬 싸게 같은 효과를 낸다.
    layer_height = detection_h or params.layer_height_mm
    if params.brim_mm > 0 and layer_height > 0:
        brim_layers = max(1, int(round(params.brim_height_mm / layer_height)))
        for i in range(min(brim_layers, n)):
            cur = clean(support[i])
            if cur.is_empty:
                continue
            # 베드에서 가장 넓고 브림 꼭대기에서 0 이 되도록 선형으로 줄인다
            grow = params.brim_mm * (1.0 - i / brim_layers)
            if grow <= 0:
                continue
            model = clean(slices[i])
            if not model.is_empty and not params.allow_internal_supports:
                model = _fill_holes(model)
            blocked = (
                model.buffer(params.xy_clearance_mm)
                if not model.is_empty else Polygon()
            )
            try:
                widened = clean(cur.buffer(grow))
                if not blocked.is_empty:
                    widened = widened.difference(blocked)
                support[i] = clean(widened)
            except Exception:
                pass  # 위상 오류가 나면 이 층은 원래대로 둔다

    if params.support_on_build_plate_only:
        base = clean(support[0])
        for i in range(n):
            cur = clean(support[i])
            if cur.is_empty or base.is_empty:
                support[i] = Polygon()
                continue
            parts = cur.geoms if hasattr(cur, "geoms") else [cur]
            keep = [p for p in parts if p.intersects(base)]
            support[i] = unary_union(keep) if keep else Polygon()

    # (3) contact = 위쪽 contact 두께 이내에서 사라지는 영역
    contact: List = [Polygon()] * n
    for i in range(n):
        cur = clean(support[i])
        above = clean(support[i + cl]) if i + cl < n else Polygon()
        contact[i] = (
            cur.difference(above) if not (cur.is_empty or above.is_empty) else cur
        )
    return support, contact
