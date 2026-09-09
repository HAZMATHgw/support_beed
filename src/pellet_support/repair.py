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
    grounded = 0
    for comp in range(ncomp):
        zmin = pts[labels == comp][:, 2].min()
        if zmin <= z0 + r0 * 1.5:
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
    query = trimesh.proximity.ProximityQuery(mesh)
    rng = np.random.default_rng(0)

    doomed = set()
    for comp in range(ncomp):
        idx = np.where(labels == comp)[0]
        cluster = pts[idx]
        if len(cluster) > sample_per_cluster:
            pick = rng.choice(len(cluster), sample_per_cluster, replace=False)
            cluster = cluster[pick]
        nearest = float(np.abs(query.signed_distance(cluster)).min())
        if nearest > reach:
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

    # 층별 '모델 + 안전여유' 를 미리 만들어 둔다. 이어 붙일 비드가 모델을
    # 파고들지 않는지 검사하는 데 쓴다.
    blocked_cache: Dict[int, object] = {}

    def blocked_at(z: float):
        k = int(max(0, min(n_det - 1, (z - z0) / det_h)))
        if k not in blocked_cache:
            model = clean(det_slices[k])
            blocked_cache[k] = (
                model.buffer(gen.xy_clearance_mm) if not model.is_empty else None
            )
        return blocked_cache[k]

    for _ in range(max_rounds):
        pts, where = _bead_points(plan, bead_h, z0)
        if len(pts) == 0:
            break
        ncomp, labels = _components(pts, pitch)
        if ncomp <= 1 and pts[:, 2].min() <= z0 + r0 * 1.5:
            break

        # 접지하지 않은 덩어리 찾기
        floating: List[int] = []
        for comp in range(ncomp):
            zmin = pts[labels == comp][:, 2].min()
            if zmin > z0 + r0 * 1.5:
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
