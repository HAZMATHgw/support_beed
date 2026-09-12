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
from trimesh.constants import tol
from trimesh.intersections import mesh_plane
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
    unique_idx, inverse = grouping.unique_rows(np.round(verts, 5))
    uniq = verts[unique_idx]
    # unique_rows가 반환한 inverse는 각 원래 끝점의 unique 인덱스다.
    # 끝점마다 다시 반올림하고 Python dict를 찾지 않아도 같은 순서와
    # 같은 대표 좌표를 그대로 유지한다.
    edges = np.asarray(inverse, dtype=np.int64).reshape(-1, 2)
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
    # mesh_multiplane는 매 층 모든 삼각형을 교차 유형별로 분류한다.
    # Z축 단면에서는 삼각형의 높이 범위가 평면에 닿는 후보만 검사해도
    # 완전히 같은 교차 선분을 얻는다. 공면/꼭짓점 처리의 허용오차까지
    # 포함하고, 원래 face 순서를 유지하여 trimesh의 교차 규칙을 재사용한다.
    vertex_dots = np.asarray(mesh.vertices[:, 2]) - z_min
    face_dots = vertex_dots[np.asarray(mesh.faces)]
    face_low = face_dots.min(axis=1)
    face_high = face_dots.max(axis=1)
    del face_dots
    normal = np.array([0.0, 0.0, 1.0])
    slices: List = []
    for height in heights:
        candidates = np.flatnonzero((face_low - height <= tol.merge)
                                    & (face_high - height >= -tol.merge))
        if len(candidates) == 0:
            slices.append(Polygon())
            continue
        lines = mesh_plane(
            mesh, normal, np.array([0.0, 0.0, z_min + height]),
            local_faces=candidates, cached_dots=vertex_dots - height,
        )
        # 수평 단면의 2D 변환은 Z 좌표를 버리는 것과 같다. 모든 층의
        # 원시 선분을 보관하지 않고 한 층씩 폴리곤으로 변환한다.
        seg = lines[:, :, :2]
        polys = [p.buffer(0) for p in segments_to_polygons(np.asarray(seg))]
        polys = [p for p in polys if not p.is_empty]
        slices.append(unary_union(polys) if polys else Polygon())
    return slices, heights + z_min
