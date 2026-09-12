# -*- coding: utf-8 -*-
"""비드 단위 연결성 복구.

서포터 '영역'이 바닥까지 이어져 있어도, 중간에 영역이 너무 좁은 층에서는
격자점이 하나도 안 들어가 비드가 안 놓인다. 그러면 영역은 연속인데 비드
사슬은 끊겨서, 위쪽 덩어리가 통째로 공중에 뜬 채 출력된다. 배 모델에서는
전체 비드의 55%가 이 상태였다.

여기서는 완성된 배치 계획을 받아 다음을 한다.

1. 비드를 노드로, 서로 닿는 비드를 간선으로 하는 그래프를 만든다.
2. 베드에도 모델에도 닿지 않는 덩어리(=공중부양)를 찾는다.
3. 그 덩어리 바닥에서 아래로 비드를 이어 붙여(stitch) 베드나 모델까지
   내려보낸다. 이어 붙이는 비드는 기존 비드와 같은 XY 격자점을 그대로 쓰므로
   평면 격자는 흐트러지지 않고, 세로로만 곧게 쌓인다(FCC 대신 수직 기둥).
   구조를 최소한으로만 건드리면서 빈 구간을 메우는 방법이다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from .params import SupportBeadParams, SupportBeadRegion, SupportGenParams
from .slicing import clean


def _bead_points(plan, bead_h: float, z0: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """(N,3) 좌표와 (레이어 인덱스, 레이어 내 인덱스) 목록을 만든다."""
    pts: List[Tuple[float, float, float]] = []
    where: List[Tuple[int, int]] = []
    for li, layer in enumerate(plan.layers):
        zc = layer.get("z_center")
        if zc is None:
            zc = layer["z_bottom"] + 0.5 * bead_h
        for bi, bead in enumerate(layer["beads"]):
            pts.append((bead["x"], bead["y"], zc))
            where.append((li, bi))
    return (np.asarray(pts, dtype=float) if pts else np.zeros((0, 3))), where


def _ground_top_z(z0: float, gen: SupportGenParams) -> float:
    """비드가 실제로 '바닥'에 닿는 높이.

    solid_first_layers 로 첫 층(들)을 beads 없는 통판으로 깔면, _bead_points
    는 그 층에서 점을 하나도 못 뽑는다. 그 상태로 z0 를 그대로 접지 기준
    삼으면, 통판 바로 위에 앉은 첫 구슬 층 전체가 '아래에 아무것도 없다'로
    오판되어 불필요한 스티칭이나 가지치기가 일어난다. 통판 꼭대기를 새
    기준으로 쓰면 그 위 첫 층도 원래 베드에 닿은 것과 똑같이 취급된다.
    """
    return z0 + gen.solid_first_layers * gen.layer_height_mm


def _components(points: np.ndarray, pitch: float):
    """서로 닿는 비드끼리 묶은 연결 덩어리 라벨."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(points)
    if n == 0:
        return 0, np.zeros(0, dtype=int)
    tree = cKDTree(points)
    pairs = tree.query_pairs(pitch * 1.02, output_type="ndarray")
    if len(pairs) == 0:
        return n, np.arange(n)
    adj = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    return connected_components(adj, directed=False)


def analyze_connectivity(plan, gen: SupportGenParams,
                         contact: SupportBeadParams, z0: float = 0.0) -> dict:
    """연결 상태를 수치로 요약한다(수리 전후 비교용)."""
    bead_h = gen.layer_height_mm
    r0 = 0.5 * contact.bead_diameter_mm
    pts, _ = _bead_points(plan, bead_h, z0)
    if len(pts) == 0:
        return {"beads": 0, "components": 0, "grounded_ratio": 1.0, "largest_ratio": 1.0}
    ncomp, labels = _components(pts, contact.pitch_mm())
    sizes = np.bincount(labels, minlength=ncomp)
    ground_z = _ground_top_z(z0, gen)
    grounded = 0
    for comp in range(ncomp):
        zmin = pts[labels == comp][:, 2].min()
        if zmin <= ground_z + r0 * 1.5:
            grounded += int(sizes[comp])
    return {
        "beads": int(len(pts)),
        "components": int(ncomp),
        "grounded_ratio": grounded / len(pts),
        "largest_ratio": float(sizes.max()) / len(pts),
    }


def prune_orphan_clusters(
    plan,
    gen: SupportGenParams,
    contact: SupportBeadParams,
    mesh,
    z0: float = 0.0,
    sample_per_cluster: int = 200,
) -> int:
    """아무것도 받치지 않는 구슬 뭉치를 지운다. 지운 구슬 수를 반환.

    베드에 얹혀 있어도 모델 근처에 가지 못하는 뭉치는 서포터 역할을 전혀
    하지 못한다. 재료만 쓰고, 인쇄 중 노즐에 걸려 떨어져 나가 다른 곳을
    망칠 수도 있다. 배 모델에서는 이런 뭉치가 약 50개(구슬 182개) 나왔다.

    '받친다'의 기준은 모델 표면까지의 최단 거리다. Z 간격과 XY 여유를 더한
    값보다 멀면 그 뭉치는 모델과 아무 관계가 없다고 본다.
    """
    import numpy as np
    import trimesh

    bead_h = gen.layer_height_mm
    pts, where = _bead_points(plan, bead_h, z0)
    if len(pts) == 0:
        return 0
    pitch = contact.pitch_mm()
    d_vert = contact.vertical_neighbor_distance_mm()
    ncomp, labels = _components(pts, max(pitch, d_vert))
    if ncomp <= 1:
        return 0

    reach = (
        gen.contact_z_gap_layers * bead_h
        + contact.bead_diameter_mm
        + gen.xy_clearance_mm
    )
    # '받치고 있다'로 볼 최대 거리.
    #
    # contact_z_gap_mm 를 빼먹으면 기준이 실제 설계 간격보다 작아져서,
    # 정상적으로 모델을 받치는 덩어리까지 '아무것도 안 받침'으로 걸러진다.
    # (덩어리가 1개일 때는 조기 반환 덕에 드러나지 않던 버그였다.)
    max_rise = (
        gen.contact_z_gap_layers * bead_h
        + gen.contact_z_gap_mm
        + contact.bead_diameter_mm * 1.5
    )
    query = trimesh.proximity.ProximityQuery(mesh)
    rng = np.random.default_rng(0)

    doomed = set()
    for comp in range(ncomp):
        idx = np.where(labels == comp)[0]
        cluster = pts[idx]

        # '모델 근처에 있는가'가 아니라 '실제로 모델을 받치고 있는가'로 본다.
        #
        # 거리만 보면 선체 옆에 붙어 있기만 한 뭉치도 통과한다. 실제로
        # 배 모델에서 7개, 7개, 5개, 5개짜리 뭉치가 아무것도 안 받치면서
        # 남아 있었다. 위로 레이를 쏴서 바로 위에 모델이 있는 구슬이 하나라도
        # 있어야 그 뭉치는 제 역할을 하는 것이다.
        origins = cluster.copy()
        origins[:, 2] += 1e-3
        holds = False
        try:
            locs, ray_idx, _ = mesh.ray.intersects_location(
                origins,
                np.tile([0.0, 0.0, 1.0], (len(origins), 1)),
                multiple_hits=False,
            )
            if len(ray_idx):
                rise = locs[:, 2] - origins[ray_idx][:, 2]
                holds = bool((rise <= max_rise).any())
        except Exception:
            # 레이 캐스팅이 안 되면 예전처럼 거리 기준으로 판단한다
            sample = cluster
            if len(sample) > sample_per_cluster:
                sample = sample[rng.choice(len(sample), sample_per_cluster,
                                           replace=False)]
            holds = float(np.abs(query.signed_distance(sample)).min()) <= reach

        if not holds:
            doomed.update(int(i) for i in idx)

    if not doomed:
        return 0

    # 뒤에서부터 지워야 인덱스가 밀리지 않는다
    removals = sorted((where[i] for i in doomed), reverse=True)
    for layer_idx, bead_idx in removals:
        del plan.layers[layer_idx]["beads"][bead_idx]
    return len(removals)


def repair_connectivity(
    plan,
    gen: SupportGenParams,
    contact: SupportBeadParams,
    det_slices,
    det_h: float,
    z0: float = 0.0,
    stride: int = 2,
    max_rounds: int = 3,
) -> int:
    """공중에 뜬 비드 덩어리를 아래로 이어 붙인다. 추가된 비드 수를 반환."""
    bead_h = gen.layer_height_mm
    if bead_h <= 0 or not plan.layers:
        return 0
    pitch = contact.pitch_mm()
    r0 = 0.5 * contact.bead_diameter_mm
    n_det = len(det_slices)
    added_total = 0
    ground_z = _ground_top_z(z0, gen)

    # 층별 '모델 + 안전여유' 를 미리 만들어 둔다. 이어 붙일 비드가 모델을
    # 파고들지 않는지 검사하는 데 쓴다.
    blocked_cache: Dict[int, object] = {}

    def blocked_at(z: float):
        k = int(max(0, min(n_det - 1, (z - z0) / det_h)))
        if k not in blocked_cache:
            # 구슬은 중심 층뿐 아니라 위아래 반지름만큼도 차지한다.
            # 그 범위의 모델을 모두 합쳐서 검사해야 한다(실측 40.9% 충돌).
            k_lo = int(max(0, min(n_det - 1, (z - r0 - z0) / det_h)))
            k_hi = int(max(0, min(n_det - 1, (z + r0 - z0) / det_h)))
            spans = [clean(det_slices[m]) for m in range(k_lo, k_hi + 1)]
            spans = [g for g in spans if not g.is_empty]
            model = unary_union(spans) if spans else clean(det_slices[k])
            # 중심점만 검사하면 안 된다. 구슬은 중심에서 반지름만큼 뻗으므로,
            # 중심이 xy_clearance 밖에 있어도 (반지름 - 여유)만큼 모델을
            # 파고든다. 노즐 5mm 에서 실측 18개가 이렇게 충돌했다.
            keep_out = max(gen.xy_clearance_mm, r0 + 0.05 * contact.bead_diameter_mm)
            blocked_cache[k] = (
                model.buffer(keep_out) if not model.is_empty else None
            )
        return blocked_cache[k]

    for _ in range(max_rounds):
        pts, where = _bead_points(plan, bead_h, z0)
        if len(pts) == 0:
            break
        ncomp, labels = _components(pts, pitch)
        if ncomp <= 1 and pts[:, 2].min() <= ground_z + r0 * 1.5:
            break

        # 접지하지 않은 덩어리 찾기
        floating: List[int] = []
        for comp in range(ncomp):
            zmin = pts[labels == comp][:, 2].min()
            if zmin > ground_z + r0 * 1.5:
                floating.append(comp)
        if not floating:
            break

        added_this_round = 0
        for comp in floating:
            idx = np.where(labels == comp)[0]
            zmin = pts[idx][:, 2].min()
            bottom = idx[pts[idx][:, 2] < zmin + pitch * 0.5]
            # 바닥 비드 전부에서 기둥을 내리면 비드가 폭증한다. 일정 간격으로
            # 솎아서 기둥 몇 개만 내린다.
            seeds = bottom[:: max(1, stride)]
            if len(seeds) == 0:
                seeds = bottom[:1]
            for si in seeds:
                x, y, z = pts[si]
                li = where[si][0]
                # 이 층 아래로 한 층씩 내려가며 비드를 놓는다
                for lj in range(li - 1, -1, -1):
                    layer = plan.layers[lj]
                    zc = layer.get("z_center")
                    if zc is None:
                        zc = layer["z_bottom"] + 0.5 * bead_h
                    blk = blocked_at(zc)
                    if blk is not None and blk.contains(Point(x, y)):
                        break  # 모델에 닿았다 = 모델 위에 얹힌 것이므로 정상 종료
                    layer["beads"].append({
                        "x": float(x), "y": float(y), "angle": 0.0,
                        "region": SupportBeadRegion.BODY,
                        "d": contact.bead_diameter_mm,
                        "stitch": True,
                    })
                    added_this_round += 1
        added_total += added_this_round
        if added_this_round == 0:
            break
    return added_total


def prune_weakly_connected(
    plan,
    gen: SupportGenParams,
    contact: SupportBeadParams,
    z0: float = 0.0,
    min_contacts: int = 3,
    max_rounds: int = 6,
    det_slices=None,
    det_h: float = 0.0,
) -> int:
    """이웃이 너무 적어 구조가 되지 못하는 구슬을 반복해서 지운다.

    이웃이 1~2개뿐인 구슬은 서포터로서 힘을 전달하지 못하고, 인쇄 중
    노즐에 걸려 떨어져 나가 다른 곳까지 망친다. 배 모델(구슬 2.5mm)에서는
    완전히 고립된 구슬 8개와 이웃 1~2개짜리 113개(12.5%)가 있었다.

    하나를 지우면 그에 기대던 구슬이 다시 약해지므로, 더 지울 것이 없을
    때까지 반복한다. 다만 베드에 얹혀 아래에서 받쳐 주는 구슬은 남긴다 —
    기둥의 맨 아래 칸은 원래 이웃이 적을 수밖에 없다.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    bead_h = gen.layer_height_mm
    r0 = 0.5 * contact.bead_diameter_mm
    radius = max(contact.pitch_mm(), contact.vertical_neighbor_distance_mm()) * 1.02
    removed_total = 0
    ground_z = _ground_top_z(z0, gen)

    for _ in range(max_rounds):
        pts, where = _bead_points(plan, bead_h, z0)
        if len(pts) < 2:
            break
        tree = cKDTree(pts)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        counts = (
            np.bincount(pairs.ravel(), minlength=len(pts))
            if len(pairs) else np.zeros(len(pts), dtype=int)
        )
        on_bed = pts[:, 2] <= ground_z + r0 * 1.5
        # 베드에 얹힌 구슬은 아래를 베드가 받쳐 주므로 기준을 낮춘다. 다만
        # 이웃이 아예 없으면 위로 아무것도 전달하지 못하므로 남길 이유가 없다.
        needed = np.where(on_bed, 1, min_contacts)
        weak = counts < needed

        # 이웃 수가 충분해도 '아래'가 비어 있으면 인쇄될 수 없다.
        # 옆으로만 붙어 있는 구슬은 허공에 놓이는 셈이다.
        if det_slices is not None and det_h > 0:
            below = np.zeros(len(pts), dtype=bool)
            if len(pairs):
                lower = pts[pairs[:, 0], 2] < pts[pairs[:, 1], 2] - 1e-9
                below[pairs[lower, 1]] = True
                below[pairs[~lower, 0]] = True
            n_det = len(det_slices)
            for i in np.where((~below) & (~on_bed) & (~weak))[0]:
                x, y, z = pts[i]
                k = int(max(0, min(n_det - 1, (z - z0 - bead_h) / det_h)))
                model = clean(det_slices[k])
                if model.is_empty or not model.contains(Point(x, y)):
                    weak[i] = True
        idx = np.where(weak)[0]
        if len(idx) == 0:
            break
        for layer_idx, bead_idx in sorted((where[i] for i in idx), reverse=True):
            del plan.layers[layer_idx]["beads"][bead_idx]
        removed_total += len(idx)
    return removed_total
