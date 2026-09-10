# -*- coding: utf-8 -*-
"""메쉬를 층별 2D 폴리곤으로 자르는 단계."""

from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh import grouping
from trimesh.intersections import mesh_multiplane
from trimesh.path.polygons import edges_to_polygons


def clean(geom):
    """비었거나 유효하지 않은 지오메트리를 안전한 형태로 정규화."""
    if geom is None or geom.is_empty:
        return Polygon()
    return geom if geom.is_valid else geom.buffer(0)


def drop_small(geom, min_area: float):
    """면적이 기준보다 작은 조각을 버린다(슬라이싱 노이즈 제거)."""
    geom = clean(geom)
    if geom.is_empty or min_area <= 0:
        return geom
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    keep = [p for p in parts if p.area >= min_area]
    return unary_union(keep) if keep else Polygon()


def segments_to_polygons(segments: np.ndarray) -> List[Polygon]:
    """(n, 2, 2) 선분 배열을 닫힌 폴리곤들로 복원한다.

    mesh_multiplane 은 폴리곤이 아니라 흩어진 선분을 준다. 부동소수점 오차로
    같은 점이 미세하게 다르게 나오므로 5자리에서 반올림해 중복을 합친 뒤,
    점 인덱스 쌍(edge)으로 바꿔 이어 붙인다.
    """
    if segments is None or len(segments) == 0:
        return []
    verts = segments.reshape(-1, 2)
    unique_idx = grouping.unique_rows(np.round(verts, 5))[0]
    uniq = verts[unique_idx]
    lookup = {tuple(np.round(v, 5)): i for i, v in enumerate(uniq)}
    try:
        edges = np.array(
            [
                [lookup[tuple(np.round(a, 5))], lookup[tuple(np.round(b, 5))]]
                for a, b in segments
            ],
            dtype=np.int64,
        )
    except KeyError:
        return []
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return []
    try:
        return list(edges_to_polygons(edges, uniq))
    except ModuleNotFoundError as exc:
        # trimesh 는 기능별 의존성을 선택 설치로 둔다. 여기서 실패하면 단면이
        # 통째로 비어서 "오버행 없음 -> 서포터 불필요" 로 둔갑하므로,
        # 삼키지 말고 무엇을 설치해야 하는지 그대로 올려보낸다.
        missing = getattr(exc, "name", None)
        if not missing:
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
            missing = m.group(1).split(".")[0] if m else None
        raise RuntimeError(
            f"단면을 폴리곤으로 잇지 못했습니다: '{missing or exc}' 패키지가 없습니다. "
            f"터미널에서 다음을 실행하세요:  pip install {missing or 'scipy'}"
        ) from exc


def slice_model(
    mesh: trimesh.Trimesh, layer_height: float, max_layers: int
) -> Tuple[List, np.ndarray]:
    """층마다 단면 폴리곤을 만든다. 각 층은 [i*h, (i+1)*h] 구간을 대표한다."""
    z_min, z_max = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    n_layers = int(math.ceil((z_max - z_min) / layer_height))
    if n_layers > max_layers:
        raise RuntimeError(
            f"레이어 수가 너무 많습니다({n_layers}). "
            f"--nozzle 을 키우거나 --max-layers 를 조정하세요."
        )
    heights = np.array([(i + 0.5) * layer_height for i in range(n_layers)])
    lines, _, _ = mesh_multiplane(
        mesh,
        np.array([0.0, 0.0, z_min]),
        np.array([0.0, 0.0, 1.0]),
        heights,
    )
    slices: List = []
    for seg in lines:
        polys = [p.buffer(0) for p in segments_to_polygons(np.asarray(seg))]
        polys = [p for p in polys if not p.is_empty]
        slices.append(unary_union(polys) if polys else Polygon())
    return slices, heights + z_min
