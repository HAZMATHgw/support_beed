# -*- coding: utf-8 -*-
"""나뭇가지형 서포터 골격(skeleton).

기존 방식("이 층의 오버행 영역을 육각 격자로 최대한 채운다")과 다르게,
여기서는 먼저 접촉점(contact point)들을 뽑고, 그 접촉점에서 베드까지
가지(branch)를 내려 병합하는 그래프를 만든다. 구슬은 이 그래프가 정해진
뒤에야, 그래프를 따라가는 방식으로 배치된다.

설계 원칙(analysis 단계에서 합의한 내용):

- 오버행 탐지(1단계, 어디에 지지가 필요한가)는 기존 `regions.py`의 로직을
  그대로 재사용한다. 바뀌는 건 그다음 "누적해서 폴리곤을 채운다"는 부분뿐이다.
- 접근 불가능한 공동(`_inaccessible_cavities`)은 접촉점을 뽑기 *전에*
  걸러낸다. 꺼낼 수 없는 곳에는애초에 가지가 자라면 안 된다.
- 병합은 거리만으로 하지 않는다(거리 + 각도 + 모델과의 여유).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from .params import SupportGenParams
from .slicing import clean


NodeKind = Literal["contact", "trunk", "root"]


@dataclass
class SupportNode:
    """골격의 노드 하나. 인덱스 기반 배열로 관리된다(포인터 대신)."""

    x: float
    y: float
    z: float
    parent: Optional[int]
    radius: float
    layer: int
    kind: NodeKind = "trunk"

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class SupportSkeleton:
    """노드 배열 + (부모,자식) 관계. children 은 조회 편의를 위한 캐시."""

    nodes: List[SupportNode] = field(default_factory=list)
    children: List[List[int]] = field(default_factory=list)

    def add_node(self, x: float, y: float, z: float, parent: Optional[int],
                radius: float, layer: int, kind: NodeKind = "trunk") -> int:
        idx = len(self.nodes)
        self.nodes.append(SupportNode(x, y, z, parent, radius, layer, kind))
        self.children.append([])
        if parent is not None:
            self.children[parent].append(idx)
        return idx

    def reparent(self, node_idx: int, new_parent_idx: int) -> None:
        """기존 노드의 parent 를 바꾼다(가지가 아래로 자라며 다음 노드를
        만들 때, '새로 만든 아래 노드'가 '기존 위쪽 끝 노드'의 parent가
        된다 — parent 는 항상 베드 쪽을 가리킨다)."""
        old_parent = self.nodes[node_idx].parent
        if old_parent is not None and node_idx in self.children[old_parent]:
            self.children[old_parent].remove(node_idx)
        self.nodes[node_idx].parent = new_parent_idx
        self.children[new_parent_idx].append(node_idx)

    def edges(self) -> List[Tuple[int, int]]:
        return [(n.parent, i) for i, n in enumerate(self.nodes) if n.parent is not None]

    def roots(self) -> List[int]:
        return [i for i, n in enumerate(self.nodes) if n.parent is None]

    def summary(self) -> str:
        n_edges = len(self.edges())
        n_leaves = sum(1 for c in self.children if not c)
        depth = self._max_depth()
        return (f"노드 {len(self.nodes)}개, 엣지 {n_edges}개, "
                f"말단(리프) {n_leaves}개, 최대 깊이 {depth}")

    def _max_depth(self) -> int:
        if not self.nodes:
            return 0
        depth = [0] * len(self.nodes)
        # 부모가 항상 인덱스상 나중에 추가되므로(위->아래로 성장), 자식 인덱스가
        # 부모보다 항상 크다는 보장은 없다(병합 시 재부모 지정). 안전하게 반복.
        changed = True
        guard = 0
        while changed and guard < len(self.nodes) + 5:
            changed = False
            guard += 1
            for i, n in enumerate(self.nodes):
                if n.parent is not None:
                    want = depth[n.parent] + 1
                    if want > depth[i]:
                        depth[i] = want
                        changed = True
        return max(depth) if depth else 0


@dataclass
class ContactPoint:
    x: float
    y: float
    z: float
    layer: int
    weight: float  # 대략적인 면적(굵기 배분에 쓸 수 있음)


def extract_contact_points(
    overhang_regions: Sequence,
    heights: Sequence[float],
    max_area_per_point: float,
    min_area: float = 0.5,
) -> List[ContactPoint]:
    """층별 오버행 폴리곤에서 접촉점을 뽑는다.

    폴리곤 하나당 기본 1개(중심점)만 뽑되, 면적이 크면
    ``max_area_per_point`` 기준으로 나눠서 여러 점을 뽑는다(넓은 오버행
    하나를 접촉점 하나로 뭉치면 그 아래 가지 하나가 감당하기엔 실제 면적이
    너무 커서, 이후 구슬 변환 단계에서 실제 접촉을 다 못 덮기 때문이다).
    """
    points: List[ContactPoint] = []
    for i, region in enumerate(overhang_regions):
        region = clean(region)
        if region.is_empty:
            continue
        parts = region.geoms if hasattr(region, "geoms") else [region]
        for part in parts:
            if part.area < min_area:
                continue
            n_pts = max(1, int(math.ceil(part.area / max_area_per_point)))
            if n_pts == 1:
                c = part.centroid
                points.append(ContactPoint(c.x, c.y, heights[i], i, part.area))
            else:
                # 면적이 큰 조각은 bbox 를 격자로 나눠 각 칸의 중심이 폴리곤
                # 안에 있으면 접촉점으로 쓴다. 완벽한 균등 분할은 아니지만,
                # "접촉점 여러 개로 나눈다"는 목적에는 충분하고 계산이 싸다.
                minx, miny, maxx, maxy = part.bounds
                side = math.sqrt(max_area_per_point)
                nx = max(1, int(math.ceil((maxx - minx) / side)))
                ny = max(1, int(math.ceil((maxy - miny) / side)))
                for ix in range(nx):
                    for iy in range(ny):
                        px = minx + (ix + 0.5) * (maxx - minx) / nx
                        py = miny + (iy + 0.5) * (maxy - miny) / ny
                        from shapely.geometry import Point as _Point
                        p = _Point(px, py)
                        if part.contains(p):
                            points.append(ContactPoint(
                                px, py, heights[i], i,
                                part.area / (nx * ny)))
                if not any(cp.layer == i for cp in points[-nx * ny:]):
                    # 격자점이 전부 폴리곤 밖(오목한 모양)이면 중심점 하나로
                    # 대체한다 — 접촉점이 하나도 안 뽑히는 것보다 낫다.
                    c = part.centroid
                    points.append(ContactPoint(c.x, c.y, heights[i], i, part.area))
    return points


def _model_blocks(x: float, y: float, model_polygon, xy_clearance: float) -> bool:
    """이 xy 위치가 모델과 너무 가까운지(파고드는지) 본다."""
    if model_polygon is None or model_polygon.is_empty:
        return False
    from shapely.geometry import Point as _Point
    return model_polygon.buffer(xy_clearance).contains(_Point(x, y))


def _find_detour(x, y, model_polygon, xy_clearance, max_step, n_dirs=16,
                 max_reach=None):
    """막힌 가지가 옆으로 돌아갈 수 있는 가장 가까운 방향을 찾는다.

    예전에는 막히면 그 층에서 그냥 대기했다. 그러면 가지가 장애물 위에
    갇혀서 아래로 못 내려가고, 병합도 못 해서 트렁크만 늘어난다.
    실제 트리 서포터는 장애물을 옆으로 돌아 내려간다.

    우회 거리는 기울임 한계(``max_step``)와 **분리해야 한다**. 기울임은
    "한 층에 얼마나 비스듬히 갈까"의 문제고, 우회는 "장애물을 벗어나려면
    얼마나 옆으로 가야 하나"의 문제라 크기가 전혀 다르다. 실측: 배 선체에
    막힌 가지가 max_lean=0.58mm 반경만 훑어서 우회로를 못 찾고 z=9.60 에서
    갇혔다(선체 벽을 넘으려면 수 mm 가 필요했다).

    16방향을 훑어 뚫린 곳 중 가장 가까운 지점을 고른다. 완전한 경로 탐색
    (A* 등)은 아니지만, '한 층 내려갈 자리'만 찾으면 되므로 이 정도로
    충분하고 훨씬 싸다.
    """
    if model_polygon is None or model_polygon.is_empty:
        return (x, y)
    if max_reach is None:
        max_reach = max(max_step * 20.0, 10.0)
    # 가까운 거리부터 시도해서, 필요 이상으로 멀리 돌아가지 않게 한다.
    radius = max(max_step, 0.5)
    while radius <= max_reach:
        for k in range(n_dirs):
            a = 2.0 * math.pi * k / n_dirs
            nx = x + radius * math.cos(a)
            ny = y + radius * math.sin(a)
            if not _model_blocks(nx, ny, model_polygon, xy_clearance):
                return (nx, ny)
        radius *= 1.6
    return None


def grow_branches(
    contact_points: List[ContactPoint],
    model_slices: Sequence,
    heights: Sequence[float],
    gen: SupportGenParams,
    step_h: float,
    merge_distance: float,
    max_merge_angle_deg: float = 50.0,
    max_branch_angle_deg: float = 25.0,
    bed_radius: float = 0.5,
    contact_z_offset: float = 0.0,
) -> SupportSkeleton:
    """접촉점에서 베드까지 가지를 내리며 병합한다.

    ``contact_z_offset`` : 접촉점을 오버행 표면 높이 그대로 두면, 그 자리에
    놓일 구슬이 표면과 같은 높이라 구 반지름만큼 모델과 수직으로 겹친다.
    XY 방향으로 아무리 밀어내도 이 겹침은 못 없앤다(실측: 배 모델에서
    관통 92개 중 60개, 65%가 접촉점 높이 ±0.5mm 안에서 발생했다). 이 만큼
    아래로 내려서 배치해야 한다(보통 0.5*접촉구슬지름 + z간격).

    핵심 발견(첫 시제품에서 실측으로 드러남): 가지가 **수직으로만** 내려가면
    시작점이 이미 가까웠던 것들 말고는 절대 병합되지 않는다. 실제 트리
    서포터가 나뭇가지처럼 보이는 이유는 가지가 아래로 내려가며 서로를 향해
    **안쪽으로 기운다**는 데 있다.

    구현 노트(정직하게 밝혀야 할 단순화):

    - 충돌 회피는 "이 xy 위치가 모델 안쪽이면 그 가지를 그 층에서는 멈추고
      다음 층에서 다시 시도"하는 나이브한 버전이다.
    - 병합은 거리+각도+모델 여유 세 조건을 모두 본다(요구사항 7번).
    """
    if not contact_points:
        return SupportSkeleton()

    skeleton = SupportSkeleton()
    z0 = min(heights)
    z_top = max(cp.z for cp in contact_points) - contact_z_offset
    max_lean = step_h * math.tan(math.radians(max_branch_angle_deg))

    active: List[dict] = []

    def find_model_at_z(z: float):
        idx = min(range(len(heights)), key=lambda i: abs(heights[i] - z))
        return clean(model_slices[idx])

    def layer_at(zz: float) -> int:
        return min(range(len(heights)), key=lambda i: abs(heights[i] - zz))

    z = z_top
    remaining = sorted(contact_points, key=lambda p: -p.z)
    ci = 0
    max_steps = int(math.ceil((z_top - z0) / step_h)) + 2
    for _ in range(max_steps):
        while ci < len(remaining) and remaining[ci].z - contact_z_offset >= z - 1e-6:
            cp = remaining[ci]
            node = skeleton.add_node(cp.x, cp.y, cp.z - contact_z_offset, None,
                                     bed_radius, cp.layer, kind="contact")
            active.append({"node": node, "x": cp.x, "y": cp.y,
                          "z": cp.z - contact_z_offset})
            ci += 1
        if not active and ci >= len(remaining):
            break

        nz = max(z0, z - step_h)
        model = find_model_at_z(nz)

        # 안쪽으로 기울이기: 각 가지 끝을 '가장 가까운 다른 가지 끝' 방향으로
        # max_lean 만큼(넘지 않게) 끌어당긴다. 전부 중심으로 당기면 실제
        # 지지점 배치와 무관하게 뭉치므로, '전체 중심'이 아니라 '가장 가까운
        # 이웃'을 목표로 삼아야 실제 트리 서포터처럼 국소적으로 합쳐진다.
        targets = []
        for i, tip in enumerate(active):
            best_d, best_j = None, None
            for j, other in enumerate(active):
                if i == j:
                    continue
                d = math.hypot(tip["x"] - other["x"], tip["y"] - other["y"])
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d > 1e-6:
                other = active[best_j]
                ux = (other["x"] - tip["x"]) / best_d
                uy = (other["y"] - tip["y"]) / best_d
                lean = min(max_lean, best_d / 2.0)  # 상대를 지나쳐 가지 않게
                targets.append((tip["x"] + ux * lean, tip["y"] + uy * lean))
            else:
                targets.append((tip["x"], tip["y"]))

        for tip, (tx, ty) in zip(active, targets):
            if _model_blocks(tx, ty, model, gen.xy_clearance_mm):
                # 기울인 위치가 막히면 기울이지 않고 제자리에서 시도한다.
                tx, ty = tip["x"], tip["y"]
            if _model_blocks(tx, ty, model, gen.xy_clearance_mm):
                # 제자리도 막혔다. 예전에는 여기서 그냥 대기했는데(그러면
                # 가지가 장애물 위에 갇혀 병합도 못 하고 트렁크만 늘어난다),
                # 실제 트리 서포터처럼 옆으로 우회해서 내려간다.
                detour = _find_detour(tip["x"], tip["y"], model,
                                      gen.xy_clearance_mm, max_lean)
                if detour is None:
                    continue  # 우회로도 없으면 이번 층은 대기
                tx, ty = detour
            new_node = skeleton.add_node(tx, ty, nz, None, bed_radius,
                                         layer_at(nz), kind="trunk")
            skeleton.reparent(tip["node"], new_node)
            tip["node"] = new_node
            tip["x"], tip["y"], tip["z"] = tx, ty, nz
        z = nz

        # 병합 검사: 거리 + 각도 + 모델 여유
        merged_flags = [False] * len(active)
        new_active: List[dict] = []
        for a in range(len(active)):
            if merged_flags[a]:
                continue
            best_b = None
            best_d = merge_distance
            for b in range(a + 1, len(active)):
                if merged_flags[b]:
                    continue
                dx = active[a]["x"] - active[b]["x"]
                dy = active[a]["y"] - active[b]["y"]
                d = math.hypot(dx, dy)
                if d < best_d:
                    best_b = b
                    best_d = d
            if best_b is not None:
                pa, pb = active[a], active[best_b]
                mx, my = (pa["x"] + pb["x"]) / 2, (pa["y"] + pb["y"]) / 2
                if not _model_blocks(mx, my, model, gen.xy_clearance_mm):
                    merge_node = skeleton.add_node(
                        mx, my, z, None, bed_radius, layer_at(z), kind="trunk")
                    skeleton.reparent(pa["node"], merge_node)
                    skeleton.reparent(pb["node"], merge_node)
                    merged_flags[a] = merged_flags[best_b] = True
                    new_active.append({"node": merge_node, "x": mx, "y": my,
                                       "z": z})
                    continue
            if not merged_flags[a]:
                new_active.append(active[a])
        active = new_active
        if z <= z0 + 1e-6:
            break

    return skeleton


def skeleton_to_bead_seeds(
    skeleton: SupportSkeleton,
    contact_diameter_mm: float,
    body_diameter_mm: float,
    model_slices: Optional[Sequence] = None,
    heights: Optional[Sequence[float]] = None,
    xy_clearance: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """엣지(부모-자식)마다 구슬 체인으로 바꾼다.

    반환값은 (x, y, z, 지름) 튜플 목록이다. 접촉점(kind="contact")에
    맞닿은 첫 구간은 인터페이스 지름을, 나머지는 몸통 지름을 쓴다.
    구슬 간격은 지름의 90%(약간 겹치게)로 잡아 사슬이 끊기지 않게 한다.

    ``model_slices``/``heights`` 를 주면 관통 보정을 한다. 성장 단계의
    충돌 회피는 노드(끝점)만 검사하므로, 두 노드 *사이* 를 잇는 체인이
    모델 모서리를 스칠 수 있다(첫 시제품 실측: 344개 중 12개가 이렇게
    관통했다). 각 구슬을 모델 바깥으로 최소 거리만 밀어낸다.
    """
    seeds: List[Tuple[float, float, float, float]] = []
    _cache: dict = {}
    for node_idx, node in enumerate(skeleton.nodes):
        if node.parent is None:
            continue
        parent = skeleton.nodes[node.parent]
        p0 = np.array([node.x, node.y, node.z])
        p1 = np.array([parent.x, parent.y, parent.z])
        length = float(np.linalg.norm(p1 - p0))
        # 노드별 굵기를 쓴다(assign_hierarchical_radii 가 매겨둔 값).
        # 부모(아래, 굵음)와 자식(위, 가늚) 사이를 보간해서 가지가
        # 자연스럽게 굵어지도록 한다.
        d_child = 2.0 * node.radius
        d_parent = 2.0 * parent.radius
        if node.kind == "contact":
            d_child = contact_diameter_mm
        if length < 1e-6:
            pts = [(node.x, node.y, node.z, d_child)]
        else:
            pitch = max(min(d_child, d_parent) * 0.9, 1e-6)
            n = max(1, int(round(length / pitch)))
            pts = []
            for i in range(n + 1):
                t = i / n
                p = p0 + (p1 - p0) * t
                dd = d_child + (d_parent - d_child) * t
                pts.append((float(p[0]), float(p[1]), float(p[2]), dd))
        if model_slices is not None and heights is not None:
            pts = [_nudge_out_of_model(x, y, z, dd, model_slices, heights,
                                       xy_clearance, _cache=_cache)
                  for x, y, z, dd in pts]
        seeds.extend(pts)
    return seeds


def _nudge_out_of_model(x, y, z, d, model_slices, heights, xy_clearance,
                        _cache: Optional[dict] = None):
    """이 구슬이 모델(+여유)을 파고들면, 가장 가까운 바깥 지점으로 밀어낸다.

    구슬은 중심 높이 하나가 아니라 위아래로 반지름만큼 뻗는다. 가장 가까운
    단일 층만 보면, 그 구슬의 z 범위 안에서 모델이 더 튀어나온 곳을 놓쳐서
    관통이 생긴다(격자 충전 방식에서 이미 한 번 겪은 버그: 실측 노즐 5mm
    에서 25~59개 관통. 여기서도 똑같이 실측 배 33%, 큐브 22% 관통했다).
    이 구슬이 차지하는 z 범위 전체의 모델을 합쳐서 검사해야 한다.

    ``_cache`` 를 주면 (lo, hi) 구간별로 union 결과를 재사용한다. 많은
    구슬이 비슷한 높이에 몰려 있어서 같은 구간을 반복 계산하는 경우가
    많다(실측: 나뭇가지 거치대에서 구슬 2,583개에 27초 — 캐시 없이는
    거의 매번 shapely union 을 새로 돌렸기 때문).
    """
    r = 0.5 * d
    lo = min(range(len(heights)), key=lambda i: abs(heights[i] - (z - r)))
    hi = min(range(len(heights)), key=lambda i: abs(heights[i] - (z + r)))
    lo, hi = min(lo, hi), max(lo, hi)
    key = (lo, hi)
    if _cache is not None and key in _cache:
        model = _cache[key]
    else:
        spanned = [clean(model_slices[i]) for i in range(lo, hi + 1)]
        spanned = [s for s in spanned if not s.is_empty]
        if not spanned:
            model = Polygon()
        else:
            from shapely.ops import unary_union as _union
            model = _union(spanned) if len(spanned) > 1 else spanned[0]
        if _cache is not None:
            _cache[key] = model
    if model.is_empty:
        return (x, y, z, d)
    from shapely.geometry import Point as _Point
    # 총 여유는 xy_clearance '만'이 아니라 구슬 자신의 반지름 + xy_clearance
    # 여야 한다. 구슬 중심이 모델에서 xy_clearance 만큼만 떨어져 있으면,
    # 반지름이 그보다 큰 구슬은 표면이 여전히 모델에 파고든다.
    # (격자 충전 방식은 center 를 놓을 영역 자체를 반지름만큼 미리 깎아
    # 둬서 이 조건을 만족시켰는데, 골격 방식으로 옮기며 이 부분을
    # 빠뜨렸다 — 배 모델 실측 관통 73개가 전부 이 원인이었다.)
    keep_out = model.buffer(xy_clearance + r + 0.15)
    p = _Point(x, y)
    if not keep_out.contains(p):
        return (x, y, z, d)
    from shapely.ops import nearest_points
    px, py = x, y
    for attempt in range(4):
        pt = Point(px, py)
        if not keep_out.contains(pt):
            return (px, py, z, d)
        # exterior 만 보면 '구멍'(hole) 안에 있을 때 엉뚱한 방향(바깥 테두리)
        # 으로 밀어서 오히려 더 깊이 들어갈 수 있다. boundary(구멍 포함)에서
        # 가장 가까운 점을 정확히 구한다.
        nearest = nearest_points(pt, keep_out.boundary)[1]
        dx, dy = nearest.x - px, nearest.y - py
        dist = math.hypot(dx, dy) or 1.0
        push = (0.05 + 0.1 * attempt) * d  # 안 되면 점점 더 세게 민다
        px = nearest.x + dx / dist * push
        py = nearest.y + dy / dist * push
    return (px, py, z, d)


def assign_hierarchical_radii(
    skeleton: SupportSkeleton,
    contact_diameter_mm: float,
    body_diameter_mm: float,
    max_trunk_diameter_mm: Optional[float] = None,
) -> None:
    """노드마다 '자기가 떠받치는 접촉점 수'에 따라 굵기를 매긴다(요구사항 10).

    트렁크는 여러 가지가 합쳐진 것이므로 굵어야 하고, 말단은 접촉점 하나만
    담당하므로 가늘어도 된다. 실제 트리 서포터도 같은 원리다.

    굵기는 부하의 제곱근에 비례시킨다 — 단면적이 부하에 비례해야 하므로
    지름은 sqrt(부하)에 비례한다. 선형으로 키우면 트렁크가 터무니없이
    굵어진다(접촉점 300개짜리 트렁크가 말단의 300배가 되어버린다).
    """
    n = len(skeleton.nodes)
    if n == 0:
        return
    if max_trunk_diameter_mm is None:
        max_trunk_diameter_mm = body_diameter_mm * 2.5

    # 각 노드가 떠받치는 접촉점 수(=자기 서브트리의 contact 노드 수).
    # parent 는 베드 쪽을 가리키므로, '자식들'이 위쪽(모델 쪽)이다.
    load = [0] * n
    order = []
    visited = [False] * n

    def walk(root: int) -> None:
        stack = [root]
        seq = []
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            seq.append(i)
            for ch in skeleton.children[i]:
                stack.append(ch)
        order.extend(reversed(seq))  # 리프부터 처리되도록

    for r in skeleton.roots():
        walk(r)
    for i in range(n):
        if not visited[i]:
            walk(i)

    for i in order:
        node = skeleton.nodes[i]
        own = 1 if node.kind == "contact" else 0
        load[i] = own + sum(load[ch] for ch in skeleton.children[i])

    for i, node in enumerate(skeleton.nodes):
        if node.kind == "contact":
            node.radius = 0.5 * contact_diameter_mm
            continue
        w = max(1, load[i])
        d = body_diameter_mm * math.sqrt(w)
        node.radius = 0.5 * min(d, max_trunk_diameter_mm)


def fix_residual_collisions(
    seeds: List[Tuple[float, float, float, float]],
    mesh,
    xy_clearance: float,
    max_iters: int = 3,
    simplify_above_faces: int = 300_000,
    chunk: int = 400,
) -> List[Tuple[float, float, float, float]]:
    """2D 근사(z 구간을 층별로 뭉쳐 보는 방식)로 못 잡는 잔여 관통을 실제
    3D 메쉬로 마지막에 한 번 더 잡는다.

    실측: 배 모델에서 2D 근사 보정 후에도 1,038개 중 11개가 남았는데,
    전부 침투 깊이가 정확히 0.086mm 로 동일했다 — 곡면 위에서 층 사이
    간격(det_h=0.4mm)으로 인한 이산화 오차였다. 2D 근사를 아무리 정교하게
    다듬어도 곡면에서는 완전히 없애기 어려우므로, 실제 메쉬 거리로 직접
    검증하고 미는 게 더 확실하다.

    메모리 관리(실측으로 밝혀낸 내용):

    trimesh 의 ``ProximityQuery.on_surface`` 는 **삼각형 수 x 조회 점 수**
    에 비례해 메모리를 쓴다. 처음엔 삼각형 수만 문제인 줄 알았는데
    (192만 각형에서 죽고 22만 각형은 정상), 10만 각형으로 줄여도 2,583점을
    한 번에 넣으면 죽었고 22만 각형이라도 2,000점씩 끊으면 멀쩡했다.
    그래서 두 가지를 같이 한다.

    1. 삼각형이 너무 많으면 단순화한 사본으로 검사한다(형상 오차는
       관통 판정에 영향을 줄 만큼 크지 않다).
    2. 조회 점을 ``chunk`` 개씩 끊어서 넣는다.
    """
    if not seeds or mesh is None:
        return seeds
    import trimesh as _trimesh

    probe = mesh
    if len(mesh.faces) > simplify_above_faces:
        try:
            probe = mesh.simplify_quadric_decimation(
                face_count=simplify_above_faces)
        except Exception:
            warnings.warn(
                f"모델이 커서(삼각형 {len(mesh.faces):,}개) 정밀 관통 보정을 "
                f"건너뜁니다. 단순화에 필요한 fast_simplification 패키지가 "
                f"없는 것으로 보입니다(pip install fast_simplification). "
                f"2D 근사 보정만 적용됩니다.",
                stacklevel=2,
            )
            return seeds

    pq = _trimesh.proximity.ProximityQuery(probe)
    out = list(seeds)
    for _ in range(max_iters):
        P = np.array([(x, y, z) for x, y, z, d in out])
        D = np.array([d for x, y, z, d in out])
        closest = np.empty_like(P)
        dist = np.empty(len(P))
        inside = np.zeros(len(P), dtype=bool)
        for s in range(0, len(P), chunk):
            e = min(s + chunk, len(P))
            c_, d_, _t = pq.on_surface(P[s:e])
            closest[s:e] = c_
            dist[s:e] = d_
            # on_surface 는 '부호 없는' 거리를 준다. 모델 **안쪽** 깊숙이
            # 있는 구슬은 표면까지 멀어서 오히려 안전해 보인다(실측: 상자
            # 한가운데 구슬이 표면까지 10mm 라 통과해 버렸다).
            # 안/밖을 따로 판정해야 한다.
            inside[s:e] = pq.signed_distance(P[s:e]) > 0
        need_r = 0.5 * D + xy_clearance
        bad = (dist < need_r) | inside
        if not bad.any():
            break
        for i in np.where(bad)[0]:
            x, y, z, d = out[i]
            cx, cy, cz = closest[i]
            if inside[i]:
                # 모델 안쪽에 있다. 가장 가까운 표면을 뚫고 나가야 하므로
                # '표면점 방향'으로 밀되, 표면을 넘어 여유까지 확보한다.
                dx, dy = cx - x, cy - y
                horiz = math.hypot(dx, dy)
                need = dist[i] + 0.5 * d + xy_clearance + 0.05
                if horiz < 0.05:
                    z_dir = 1.0 if cz > z else -1.0
                    out[i] = (x, y, z + z_dir * need, d)
                else:
                    out[i] = (x + dx / horiz * need, y + dy / horiz * need, z, d)
                continue
            dx, dy = x - cx, y - cy
            horiz = math.hypot(dx, dy)
            push = (need_r[i] - dist[i]) + 0.05
            if horiz < 0.05:
                # 표면점이 거의 정확히 위/아래에 있다(실측: 배 모델에서
                # 남은 11개가 전부 수평거리 0.000, 수직 1.164mm 였다).
                # 이런 경우 아무리 옆으로 밀어도 절대 못 피한다 — 겹치는
                # 방향(z)으로 밀어야 한다.
                z_dir = 1.0 if z > cz else -1.0
                out[i] = (x, y, z + z_dir * push, d)
            else:
                nx = x + dx / horiz * push
                ny = y + dy / horiz * push
                out[i] = (nx, ny, z, d)
    return out

