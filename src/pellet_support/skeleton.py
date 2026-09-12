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
from shapely.geometry import Point, Polygon, box

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
        # 루트에서 한 번만 내려가면 O(노드 수). 전체 배열을 깊이만큼
        # 반복 갱신하던 방식은 작은 비드/높은 모델에서 비용이 커진다.
        deepest = 0
        stack = [(idx, 0) for idx in self.roots()]
        seen = set()
        while stack:
            idx, depth = stack.pop()
            if idx in seen:
                continue
            seen.add(idx)
            deepest = max(deepest, depth)
            stack.extend((child, depth + 1) for child in self.children[idx])
        return deepest


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
    contact_spacing_mm: Optional[float] = None,
) -> List[ContactPoint]:
    """오버행을 일정 간격으로 샘플링하고 연속된 경사면의 반복 접점을 줄인다.

    각 격자 셀과 실제 폴리곤의 교집합 안에서 점을 선택한다. 오목한 면과
    좁은 섬도 누락하지 않는다. 이전 층과 이어지는 조각끼리만 접점을
    공유하므로, 빈 층을 사이에 둔 별도 선반의 접점은 사라지지 않는다.
    """
    spacing = contact_spacing_mm if contact_spacing_mm is not None else math.sqrt(max_area_per_point)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("접점 간격은 유한한 양수여야 합니다.")
    if len(overhang_regions) != len(heights):
        raise ValueError("오버행 영역과 높이 수가 다릅니다.")
    points: List[ContactPoint] = []
    previous = []
    for i, region in enumerate(overhang_regions):
        region = clean(region)
        if region.is_empty:
            previous = []
            continue
        parts = region.geoms if hasattr(region, "geoms") else [region]
        current = []
        dz = abs(heights[i] - heights[i - 1]) if i else 0.0
        for part in parts:
            if part.area <= 0 or part.area < min_area:
                continue
            recent = []
            seen = set()
            for old_part, old_points in previous:
                if part.distance(old_part) <= dz + 1e-8:
                    for cp in old_points:
                        if heights[i] - cp.z < spacing * 0.75 and id(cp) not in seen:
                            recent.append(cp)
                            seen.add(id(cp))
            # 공간 해시로 가까운 점만 검사하여 큰 면에서의 전체 비교를 피한다.
            buckets = {}
            def add_bucket(cp):
                key = (math.floor(cp.x / spacing), math.floor(cp.y / spacing))
                buckets.setdefault(key, []).append(cp)
            for cp in recent:
                add_bucket(cp)
            minx, miny, maxx, maxy = part.bounds
            for ix in range(math.floor(minx / spacing), math.ceil(maxx / spacing)):
                for iy in range(math.floor(miny / spacing), math.ceil(maxy / spacing)):
                    cell = part.intersection(box(ix * spacing, iy * spacing,
                                                 (ix + 1) * spacing, (iy + 1) * spacing))
                    if cell.is_empty or cell.area <= 1e-12:
                        continue
                    c = cell.centroid
                    if not cell.covers(c):
                        c = cell.representative_point()
                    near = (cp for bx in range(ix - 1, ix + 2)
                            for by in range(iy - 1, iy + 2)
                            for cp in buckets.get((bx, by), []))
                    if any((cp.x - c.x) ** 2 + (cp.y - c.y) ** 2
                           + (cp.z - heights[i]) ** 2 < (spacing * 0.75) ** 2
                           for cp in near):
                        continue
                    cp = ContactPoint(c.x, c.y, heights[i], i, cell.area)
                    points.append(cp)
                    recent.append(cp)
                    add_bucket(cp)
            current.append((part, recent))
        previous = current
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
    bed_z: Optional[float] = None,
    contact_radius: Optional[float] = None,
) -> SupportSkeleton:
    """도달 가능한 아래층에서만 병합하는, 각도 제한이 있는 탐욕적 트리 성장.

    다음 층까지의 높이 차와 기울기로 두 도달 원을 만들고 교집합에서
    합류한다. 엣지 전체의 반지름/모델 충돌을 검사하며, 통과할 수 없는
    벽을 건너뛰지 않는다. 베드 또는 허용된 모델 윗면에 도달하지 못한
    나무는 통째로 제거하고 누락된 접점 수를 경고한다.
    """
    from bisect import bisect_left
    from .tree_collision import SliceCollision

    if step_h <= 0 or merge_distance <= 0 or bed_radius <= 0:
        raise ValueError("성장 간격, 병합 거리, 가지 반지름은 양수여야 합니다.")
    if not (0 <= max_branch_angle_deg < 90 and 0 < max_merge_angle_deg < 180):
        raise ValueError("가지/병합 각도가 유효 범위를 벗어났습니다.")
    skeleton = SupportSkeleton()
    if not contact_points:
        return skeleton
    if len(heights) == 0:
        raise ValueError("모델 슬라이스 높이가 필요합니다.")
    collision = SliceCollision(model_slices, heights, gen.xy_clearance_mm)
    bed = float(heights[0]) if bed_z is None else bed_z
    tip_radius = bed_radius if contact_radius is None else contact_radius
    lean = math.tan(math.radians(max_branch_angle_deg))
    merge_lean = min(lean, math.tan(math.radians(max_merge_angle_deg * 0.5)))
    active = []
    grounded = set()
    remaining = sorted(contact_points, key=lambda p: (-p.z, p.x, p.y))
    ci = 0
    z = remaining[0].z - contact_z_offset

    def layer_at(zz):
        return min(len(heights) - 1, max(0, bisect_left(heights, zz)))

    def xyz(node):
        return (node.x, node.y, node.z)

    def radius_for(load):
        return bed_radius * min(2.5, math.sqrt(load)) if gen.adaptive_bead_size else bed_radius

    def attach(tips, x, y, zz, radius):
        if gen.max_beads is not None and len(skeleton.nodes) >= gen.max_beads:
            from .validation import guard_bead_count
            guard_bead_count(len(skeleton.nodes) + 1, gen.max_beads)
        idx = skeleton.add_node(x, y, zz, None, radius, layer_at(zz))
        for tip in tips:
            skeleton.reparent(tip["node"], idx)
        if zz - radius <= bed + 1e-7:
            skeleton.nodes[idx].kind = "root"
            grounded.add(idx)
            return None
        return {"node": idx, "load": sum(t["load"] for t in tips)}

    # 공간 해시로 병합 후보를 국소적으로 찾는다. 먼 가지끼리 전부 비교하지 않는다.
    def nearby_pairs(tips):
        buckets = {}
        pairs = []
        for i, tip in enumerate(tips):
            node = skeleton.nodes[tip["node"]]
            key = (math.floor(node.x / merge_distance), math.floor(node.y / merge_distance))
            for bx in range(key[0] - 1, key[0] + 2):
                for by in range(key[1] - 1, key[1] + 2):
                    for j in buckets.get((bx, by), []):
                        other = skeleton.nodes[tips[j]["node"]]
                        d = math.hypot(node.x - other.x, node.y - other.y)
                        if d <= merge_distance:
                            pairs.append((d, j, i))
            buckets.setdefault(key, []).append(i)
        return sorted(pairs)

    while z >= bed + bed_radius - 1e-7:
        while ci < len(remaining) and remaining[ci].z - contact_z_offset >= z - 1e-7:
            cp = remaining[ci]
            ci += 1
            zz = cp.z - contact_z_offset
            if zz < bed + tip_radius or not collision.sphere_clear(cp.x, cp.y, zz, tip_radius):
                continue
            idx = skeleton.add_node(cp.x, cp.y, zz, None, tip_radius, cp.layer, kind="contact")
            if zz - tip_radius <= bed + 1e-7:
                grounded.add(idx)
            else:
                active.append({"node": idx, "load": 1})
        nz = max(bed + bed_radius, z - step_h)
        pairs = nearby_pairs(active)
        consumed = set()
        next_active = []
        nearest = {}
        for distance, a, b in pairs:
            nearest.setdefault(a, b)
            nearest.setdefault(b, a)
            if a in consumed or b in consumed:
                continue
            pa, pb = active[a], active[b]
            na, nb = skeleton.nodes[pa["node"]], skeleton.nodes[pb["node"]]
            desired = radius_for(pa["load"] + pb["load"])
            for radius in sorted({bed_radius, desired}, reverse=True):
                zz = max(nz, bed + radius)
                if zz >= min(na.z, nb.z) - 1e-7:
                    continue
                ra, rb = (na.z - zz) * merge_lean, (nb.z - zz) * merge_lean
                lo, hi = max(0.0, distance - rb), min(distance, ra)
                if lo > hi + 1e-8:
                    continue
                along = min(hi, max(lo, distance * pb["load"] / (pa["load"] + pb["load"])))
                t = along / distance if distance > 1e-9 else 0.5
                x, y = na.x + (nb.x - na.x) * t, na.y + (nb.y - na.y) * t
                target = (x, y, zz)
                if not (collision.edge_clear(xyz(na), target, na.radius, radius)
                        and collision.edge_clear(xyz(nb), target, nb.radius, radius)):
                    continue
                tip = attach([pa, pb], x, y, zz, radius)
                if tip is not None:
                    next_active.append(tip)
                consumed.update((a, b))
                break

        for i, tip in enumerate(active):
            if i in consumed:
                continue
            node = skeleton.nodes[tip["node"]]
            moved = False
            desired = radius_for(tip["load"])
            for radius in sorted({bed_radius, min(node.radius, desired), desired}, reverse=True):
                zz = max(nz, bed + radius)
                if zz >= node.z - 1e-7:
                    continue
                reach = (node.z - zz) * lean
                targets = []
                if i in nearest:
                    other = skeleton.nodes[active[nearest[i]]["node"]]
                    dx, dy = other.x - node.x, other.y - node.y
                    d = math.hypot(dx, dy)
                    t = min(reach / d, 0.5) if d > 1e-9 else 0.0
                    targets.append((node.x + dx * t, node.y + dy * t))
                targets.append(node.xy)
                # 우회도 도달 반경 안에서만 시도한다. 벽 너머로 건너뛰지 않는다.
                for scale in (1.0, 0.5):
                    if reach > 1e-9:
                        targets.extend((node.x + reach * scale * math.cos(k * math.pi / 8),
                                        node.y + reach * scale * math.sin(k * math.pi / 8))
                                       for k in range(16))
                for x, y in targets:
                    if collision.edge_clear(xyz(node), (x, y, zz), node.radius, radius):
                        nxt = attach([tip], x, y, zz, radius)
                        if nxt is not None:
                            next_active.append(nxt)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                continue
            # 모델 위 착지가 허용된 경우에만 바로 아래의 면에서 멈춘다.
            if not gen.support_on_build_plate_only:
                lo, hi = max(bed + node.radius, nz), node.z
                if lo < hi and not collision.sphere_clear(node.x, node.y, lo, node.radius):
                    for _ in range(24):
                        mid = (lo + hi) * 0.5
                        if collision.sphere_clear(node.x, node.y, mid, node.radius):
                            hi = mid
                        else:
                            lo = mid
                    k = max(0, min(len(heights) - 1, bisect_left(heights, hi - node.radius) - 1))
                    if clean(model_slices[k]).covers(Point(node.x, node.y)):
                        if hi < node.z - 1e-7 and collision.edge_clear(xyz(node), (node.x, node.y, hi), node.radius, node.radius):
                            landed = attach([tip], node.x, node.y, hi, node.radius)
                            idx = landed["node"] if landed is not None else len(skeleton.nodes) - 1
                        else:
                            idx = tip["node"]
                        grounded.add(idx)
                        # 접점 하나만 있는 나무는 contact 종류를 유지한다.
                        if skeleton.nodes[idx].kind != "contact":
                            skeleton.nodes[idx].kind = "root"
            # 경로가 없으면 대기하거나 순간이동하지 않고 이 나무를 제거한다.
        active = next_active
        if nz >= z - 1e-8:
            break
        z = nz

    keep = set()
    stack = list(grounded)
    while stack:
        idx = stack.pop()
        if idx not in keep:
            keep.add(idx)
            stack.extend(skeleton.children[idx])
    result = SupportSkeleton()
    mapping = {}
    for old in sorted(keep):
        n = skeleton.nodes[old]
        mapping[old] = result.add_node(n.x, n.y, n.z, None, n.radius, n.layer, n.kind)
    for old, new in mapping.items():
        parent = skeleton.nodes[old].parent
        if parent in mapping:
            result.reparent(new, mapping[parent])
    lost = len(contact_points) - sum(n.kind == "contact" for n in result.nodes)
    if lost:
        warnings.warn(f"트리 접점 {len(contact_points)}개 중 {lost}개는 각도/충돌/접지 조건을 "
                      "만족하는 경로가 없어 제외했습니다. 미리보기에서 지지 누락을 확인하세요.", stacklevel=2)
    return result


def skeleton_to_bead_seeds(
    skeleton: SupportSkeleton,
    contact_diameter_mm: float,
    body_diameter_mm: float,
    model_slices: Optional[Sequence] = None,
    heights: Optional[Sequence[float]] = None,
    xy_clearance: float = 0.0,
    max_beads: Optional[int] = None,
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
    seeds = {}
    tolerance = min(contact_diameter_mm, body_diameter_mm) * 1e-7
    if tolerance <= 0:
        raise ValueError("구슬 지름은 양수여야 합니다.")

    def add_seed(seed):
        key = tuple(round(v / tolerance) for v in seed[:3])
        old = seeds.get(key)
        if old is None or seed[3] > old[3]:
            seeds[key] = seed
        if max_beads is not None and len(seeds) > max_beads:
            from .validation import guard_bead_count
            guard_bead_count(len(seeds), max_beads)

    _cache: dict = {}
    for node_idx, node in enumerate(skeleton.nodes):
        if node.parent is None:
            # 단독 접지 접점도 하나의 유효한 구슬이다.
            if not skeleton.children[node_idx]:
                add_seed((node.x, node.y, node.z, 2.0 * node.radius))
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
            # round 는 간격을 구 지름보다 크게 만들어 사슬을 끊을 수 있다.
            n = max(1, int(math.ceil(length / pitch)))
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
        for seed in pts:
            add_seed(seed)
    return list(seeds.values())


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

