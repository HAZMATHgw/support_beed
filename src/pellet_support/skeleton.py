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
    #: 이 노드에서 가지가 베드에 닿았다(뿌리).
    on_bed: bool = False
    #: 베드까지 못 가고 모델 윗면에 내려앉았다.
    on_model: bool = False

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class SupportSkeleton:
    """노드 배열 + (부모,자식) 관계. children 은 조회 편의를 위한 캐시."""

    nodes: List[SupportNode] = field(default_factory=list)
    children: List[List[int]] = field(default_factory=list)
    #: 베드에 너무 가까워 구슬이 들어갈 틈이 없어 건너뛴 접촉점 수.
    skipped_contacts: int = 0
    #: 벽과 너무 붙어 있어 주변에 구슬을 둘 빈 자리가 없던 접촉점 수.
    blocked_contacts: int = 0
    #: 건너뛴 접촉점 좌표(진단용).
    dropped_points: List[Tuple[float, float, float]] = field(default_factory=list)

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



class _SliceField:
    """탐지 슬라이스를 '구슬 중심을 둘 수 없는 영역'과 '베드까지 못 가는
    영역'으로 미리 바꿔 둔다.

    예전 성장 코드는 매 층 "지금 이 점이 모델 안인가"만 봤다. 그러면
    가지가 모델 위(예: 받침판, 아래로 내려오는 다른 가지)에 올라탄 뒤에야
    막힌 걸 알게 되고, 그때는 기울기 한계 안에서 빠져나갈 길이 없다.
    그래서 옛 코드는 한 층에 수 mm~11mm 를 옆으로 '순간이동'(우회)하거나,
    같은 높이에서 수평으로 병합했는데, 이게 전부 공중에 뜬 수평 구슬
    사슬이 됐다(실측: 엣지 35~43% 가 사실상 수평, 구슬 14~35% 가 받침 없이
    허공에 찍힘).

    - ``keep_out[i]`` : i 번째 슬라이스 높이에 구슬 중심을 두면 구슬(반지름
      ``r``)이 모델+여유를 파고드는 영역. 구슬이 위아래로 차지하는 높이
      범위의 슬라이스를 전부 합쳐서 본다.
    - ``doomed[i]`` : 여기서 출발하면 기울기 한계(층당 ``lean``) 안에서는
      어떻게 움직여도 결국 모델에 막히는 영역(Cura 의 avoidance 와 같은
      개념). 아래에서부터 누적한다:
      ``doomed[i] = keep_out[i] ∪ erode(doomed[i-1], lean)``.
    """

    def __init__(self, model_slices, heights, r: float, clearance: float,
                 lean_per_slice: float, top_gap: float = 0.3):
        import shapely
        from shapely.ops import unary_union

        self.heights = list(heights)
        self.n = len(self.heights)
        self.det_h = (self.heights[1] - self.heights[0]) if self.n > 1 else 1.0
        self.bed_z = self.heights[0] - 0.5 * self.det_h
        # 고해상도 모델(삼각형 190만 개급)은 단면 꼭짓점이 수천 개라 buffer 가
        # 느리다. 구슬 반지름의 10% 오차는 여유(clearance)에 묻히므로 먼저
        # 단순화하고, 같은 (슬라이스, 반경) buffer 는 한 번만 계산한다.
        slices = [clean(clean(s).simplify(0.1 * r)) for s in model_slices]
        a = int(math.ceil(r / self.det_h))
        buf_cache: dict = {}
        self.keep_out = []
        self.doomed = []
        prev = Polygon()
        for i in range(self.n):
            lo, hi = max(0, i - a), min(self.n - 1, i + a)
            # 구슬 단면은 높이에 따라 sqrt(r^2 - dz^2) 로 줄어든다. 모든 층을
            # r+여유로 똑같이 부풀리면, 경사진 오버행 바로 아래의 접촉 구슬이
            # 제자리에서 한 발짝도 못 움직인다(실측: 89개 가지가 출발도 못 함).
            # 중심보다 위쪽 층은 z 간격(top_gap)만, 아래쪽은 XY 여유를 쓴다.
            parts = []
            for j in range(lo, hi + 1):
                if slices[j].is_empty:
                    continue
                dz = self.heights[j] - self.heights[i]
                # 슬라이스 j 는 높이 ±det_h/2 두께를 가진다. 구슬과 수직으로
                # 겹치지 않는 층은 볼 필요가 없다(겹치는 가장 가까운 높이 기준).
                gap_z = abs(dz) - 0.5 * self.det_h
                if gap_z >= r:
                    continue
                rr = math.sqrt(r * r - max(0.0, gap_z) ** 2)
                c = clearance if dz <= 0 else min(clearance, top_gap)
                key = (j, round(rr + c, 4))
                if key not in buf_cache:
                    buf_cache[key] = slices[j].buffer(rr + c, quad_segs=4)
                parts.append(buf_cache[key])
            for key in [kk for kk in buf_cache if kk[0] < lo]:
                del buf_cache[key]
            if parts:
                k = clean(unary_union(parts).simplify(0.05 * r))
            else:
                k = Polygon()
            if not prev.is_empty:
                eroded = clean(prev.buffer(-lean_per_slice, quad_segs=4))
                d = clean(unary_union([k, eroded])) if not k.is_empty else eroded
            else:
                d = k
            d = clean(d.simplify(0.02 * r)) if not d.is_empty else d
            shapely.prepare(k)
            shapely.prepare(d)
            self.keep_out.append(k)
            self.doomed.append(d)
            prev = d

    def index(self, z: float) -> int:
        i = int(round((z - self.heights[0]) / self.det_h))
        return max(0, min(self.n - 1, i))

    def blocked(self, x: float, y: float, z: float) -> bool:
        import shapely
        g = self.keep_out[self.index(z)]
        return (not g.is_empty) and bool(shapely.contains_xy(g, x, y))

    def is_doomed(self, x: float, y: float, z: float) -> bool:
        import shapely
        g = self.doomed[self.index(z)]
        return (not g.is_empty) and bool(shapely.contains_xy(g, x, y))


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
    bead_radius: Optional[float] = None,
) -> SupportSkeleton:
    """접촉점에서 베드(또는 모델 윗면)까지 가지를 내리며 병합한다.

    **모든 이동은 기울기 한계 안에서만 일어난다.** 이것이 예전 구현과의
    핵심 차이다. 한 단계(``step_h``) 내려갈 때 옆으로 움직일 수 있는 거리는
    ``step_h * tan(max_branch_angle_deg)`` 를 절대 넘지 않는다.

    - 병합: 두 가지 끝이 '한 단계 안에 서로 만날 수 있을 만큼' 가까워지면
      **다음 층의 한 점**에서 만난다(V 자). 예전처럼 같은 높이에서 옆으로
      잇는 수평 다리는 만들지 않는다 — 수평 구슬 사슬은 아래가 비어 있어
      출력이 안 된다.
    - 끌림: 병합 반경(``merge_distance``) 안의 가장 가까운 가지 쪽으로
      기운다. 짐(떠받치는 접촉점 수)이 무거운 가지는 덜 움직이고 가벼운
      가지가 더 움직여서, 큰 트렁크는 곧게 서고 잔가지가 트렁크로 모인다.
    - 회피: ``_SliceField.doomed`` 로 '베드까지 갈 수 있는 곳'을 미리 알고
      그 안에서만 움직인다. 이미 갈 곳이 없는 가지(예: 받침판 바로 위의
      오버행)는 모델 윗면에 내려앉는다(``on_model=True``).

    ``contact_z_offset`` 은 접촉 구슬이 오버행 표면을 파고들지 않도록 접촉점
    아래로 내리는 양이다(보통 0.5*접촉구슬지름 + z간격).
    """
    if not contact_points:
        return SupportSkeleton()
    from scipy.spatial import cKDTree
    from .tree_collision import SliceCollision

    r_bead = bead_radius if bead_radius is not None else bed_radius
    tan_a = math.tan(math.radians(max_branch_angle_deg))
    heights = list(heights)
    det_h = (heights[1] - heights[0]) if len(heights) > 1 else step_h
    field = _SliceField(model_slices, heights, r_bead, gen.xy_clearance_mm,
                        lean_per_slice=0.9 * det_h * tan_a)
    collision = SliceCollision(model_slices, heights, gen.xy_clearance_mm)
    z_end = field.bed_z + 0.8 * r_bead  # 베드에 살짝 눌러 앉힌 구슬 중심

    skeleton = SupportSkeleton()
    remaining = sorted(contact_points, key=lambda p: -p.z)
    active: List[dict] = []
    ci = 0
    z = max(cp.z for cp in contact_points) - contact_z_offset

    def valid(x, y, zz, allow_doomed):
        if field.blocked(x, y, zz):
            return False
        return allow_doomed or not field.is_doomed(x, y, zz)

    def best_position(tip, want_x, want_y, nz, lean):
        """기울기 한계(원판 반경 lean) 안에서 want 에 가장 가까운 유효 위치.

        1순위: 베드까지 갈 수 있는 곳. 2순위: 막히지만 않은 곳.
        우회도 사용자가 지정한 기울기 안에서만 찾고 연결 구간을 검사한다.
        """
        doomed_now = field.is_doomed(tip["x"], tip["y"], tip["z"])
        tries = [(1.0, False), (1.0, True)] if not doomed_now else [(1.0, True)]
        def clear_edge(x, y):
            return collision.edge_clear((tip["x"], tip["y"], tip["z"]),
                                        (x, y, nz), r_bead, r_bead)
        for mult, allow_doomed in tries:
            lim = lean * mult
            if math.hypot(want_x - tip["x"], want_y - tip["y"]) <= lim + 1e-9 \
                    and valid(want_x, want_y, nz, allow_doomed) and clear_edge(want_x, want_y):
                return want_x, want_y
            best, best_d = None, None
            for frac in (0.0, 0.34, 0.67, 1.0):
                rr = lim * frac
                dirs = 1 if frac == 0.0 else 16
                for k in range(dirs):
                    ang = 2.0 * math.pi * k / dirs
                    cx = tip["x"] + rr * math.cos(ang)
                    cy = tip["y"] + rr * math.sin(ang)
                    if not valid(cx, cy, nz, allow_doomed):
                        continue
                    if not clear_edge(cx, cy):
                        continue
                    dd = math.hypot(cx - want_x, cy - want_y)
                    if best_d is None or dd < best_d:
                        best, best_d = (cx, cy), dd
            if best is not None:
                return best
        return None

    def land(tip):
        """막힌 가지를 바로 아래 모델 면까지 내려앉힌다(슬래브 근사)."""
        lo, hi = z_end, tip["z"]
        # 수직으로 훑어 처음 막히는 구간을 찾는다. 장애물을 건너뛰지 않는다.
        while hi > z_end:
            below = max(z_end, hi - min(det_h, step_h))
            if not collision.edge_clear((tip["x"], tip["y"], hi),
                                        (tip["x"], tip["y"], below), r_bead, r_bead):
                lo = below
                break
            hi = below
        if hi <= z_end + 1e-8:
            low = z_end
            on_bed = True
        else:
            for _ in range(24):
                mid = (lo + hi) * 0.5
                if collision.sphere_clear(tip["x"], tip["y"], mid, r_bead):
                    hi = mid
                else:
                    lo = mid
            low, on_bed = hi, False
        if low < tip["z"] - 1e-8:
            node = skeleton.add_node(tip["x"], tip["y"], low, None, bed_radius,
                                     field.index(low), kind="trunk")
            skeleton.reparent(tip["node"], node)
            tip["node"] = node
        root = skeleton.nodes[tip["node"]]
        root.on_bed = on_bed
        root.on_model = not on_bed

    max_iter = int(math.ceil((z - z_end) / (0.25 * step_h))) + 10
    for _ in range(max_iter):
        nz = max(z_end, z - step_h)
        # 이번 단계 사이(nz, z]에 시작하는 접촉점을 제 높이에 추가한다.
        while ci < len(remaining) and remaining[ci].z - contact_z_offset > nz + 1e-9:
            cp = remaining[ci]
            ci += 1
            cz = cp.z - contact_z_offset
            px, py = cp.x, cp.y
            if field.blocked(px, py, cz) or field.is_doomed(px, py, cz):
                # 벽에 붙은 오버행(예: 가지가 줄기에 붙는 모서리)에서 뽑힌
                # 접촉점은 제자리가 이미 벽 여유 안이거나, 거기서 내려가면
                # 벽에 막힌다. 가까운 곳에 베드까지 갈 수 있는 빈 자리가 있으면
                # 그리로 옮겨서 시작한다(예전에는 구슬을 하나씩 밀어냈다).
                free_spot = None
                good_spot = None
                blocked_here = field.blocked(px, py, cz)
                # 막힌 게 아니라 '내려가면 막히는' 자리면 오버행 밑을 벗어나지
                # 않도록 조금만 옮겨 본다(못 찾으면 제자리에서 모델 위에 얹힌다).
                reach = (r_bead + gen.xy_clearance_mm + 2.0 * r_bead
                         if blocked_here else r_bead)
                for frac in (0.25, 0.5, 0.75, 1.0):
                    for k in range(16):
                        a_ = 2.0 * math.pi * k / 16
                        qx = cp.x + reach * frac * math.cos(a_)
                        qy = cp.y + reach * frac * math.sin(a_)
                        if field.blocked(qx, qy, cz):
                            continue
                        if free_spot is None:
                            free_spot = (qx, qy)
                        if not field.is_doomed(qx, qy, cz):
                            good_spot = (qx, qy)
                            break
                    if good_spot:
                        break
                if good_spot is not None:
                    px, py = good_spot
                elif field.blocked(px, py, cz):
                    if free_spot is None:
                        skeleton.blocked_contacts += 1
                        skeleton.dropped_points.append((cp.x, cp.y, cp.z))
                        continue
                    px, py = free_spot
            node = skeleton.add_node(px, py, cz, None, bed_radius, cp.layer,
                                     kind="contact")
            active.append({"node": node, "x": px, "y": py, "z": cz,
                           "load": 1, "ztop": cp.z})
        if nz <= z_end + 1e-9:
            # 베드 높이보다 낮게 시작해야 하는 접촉점은 구슬이 들어갈 틈이 없다.
            while ci < len(remaining):
                cp = remaining[ci]
                ci += 1
                if cp.z - contact_z_offset > z_end - 0.5 * r_bead and \
                        not field.blocked(cp.x, cp.y, z_end):
                    node = skeleton.add_node(cp.x, cp.y, z_end, None, bed_radius,
                                             cp.layer, kind="contact")
                    skeleton.nodes[node].on_bed = True
                else:
                    skeleton.skipped_contacts += 1
                    skeleton.dropped_points.append((cp.x, cp.y, cp.z))
        if not active:
            if ci >= len(remaining):
                break
            z = max(nz, remaining[ci].z - contact_z_offset)
            continue

        leans = [max(0.0, t["z"] - nz) * tan_a for t in active]
        pts = np.array([(t["x"], t["y"]) for t in active])
        tree = cKDTree(pts) if len(active) > 1 else None
        merged = [False] * len(active)
        new_active: List[dict] = []

        # 1) 병합: 한 단계 안에 같은 점으로 모일 수 있는 쌍 (가까운 쌍부터)
        if tree is not None:
            reach = min(merge_distance, 2.0 * max(leans)) if leans else 0.0
            pairs = sorted(
                ((math.hypot(*(pts[a] - pts[b])), a, b)
                 for a, b in tree.query_pairs(reach)),
                key=lambda t: t[0])
            for d, a, b in pairs:
                if merged[a] or merged[b]:
                    continue
                ta, tb = active[a], active[b]
                wa, wb = ta["load"], tb["load"]
                mx = (ta["x"] * wa + tb["x"] * wb) / (wa + wb)
                my = (ta["y"] * wa + tb["y"] * wb) / (wa + wb)
                if math.hypot(mx - ta["x"], my - ta["y"]) > leans[a] + 1e-9 or \
                        math.hypot(mx - tb["x"], my - tb["y"]) > leans[b] + 1e-9:
                    continue
                both_doomed = (field.is_doomed(ta["x"], ta["y"], ta["z"]) and
                               field.is_doomed(tb["x"], tb["y"], tb["z"]))
                if not valid(mx, my, nz, allow_doomed=both_doomed):
                    continue
                if not all(collision.edge_clear((t["x"], t["y"], t["z"]),
                                                 (mx, my, nz), r_bead, r_bead)
                           for t in (ta, tb)):
                    continue
                node = skeleton.add_node(mx, my, nz, None, bed_radius,
                                         field.index(nz), kind="trunk")
                skeleton.reparent(ta["node"], node)
                skeleton.reparent(tb["node"], node)
                merged[a] = merged[b] = True
                new_active.append({"node": node, "x": mx, "y": my, "z": nz,
                                   "load": wa + wb,
                                   "ztop": max(ta["ztop"], tb["ztop"])})

        # 2) 나머지: 가까운 가지 쪽으로 기울며 한 단계 내려간다
        for i, tip in enumerate(active):
            if merged[i]:
                continue
            want_x, want_y = tip["x"], tip["y"]
            if tree is not None:
                # 끌림 반경은 '지금까지 내려온 높이로 옆으로 갈 수 있는 거리'의
                # 절반까지 넓힌다. 고정 반경(구슬 지름 6배)만 쓰면 작은 구슬에서
                # 가지들이 끝까지 안 모여 트렁크가 수십 개 따로 선다(실측: 노즐
                # 1mm 거치대에서 트렁크 80개).
                attract = max(merge_distance,
                              0.5 * (tip["ztop"] - contact_z_offset - tip["z"]) * tan_a)
                dists, idxs = tree.query(pts[i], k=min(len(active), 6))
                for dj, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                    if j == i or dj < 1e-9 or dj > attract:
                        continue
                    other = active[j]
                    share = other["load"] / (tip["load"] + other["load"])
                    move = min(leans[i], dj * share)
                    want_x = tip["x"] + (other["x"] - tip["x"]) / dj * move
                    want_y = tip["y"] + (other["y"] - tip["y"]) / dj * move
                    break
            pos = best_position(tip, want_x, want_y, nz, leans[i])
            if pos is None:
                land(tip)
                continue
            node = skeleton.add_node(pos[0], pos[1], nz, None, bed_radius,
                                     field.index(nz), kind="trunk")
            skeleton.reparent(tip["node"], node)
            tip.update(node=node, x=pos[0], y=pos[1], z=nz)
            new_active.append(tip)

        active = new_active
        z = nz
        if nz <= z_end + 1e-9:
            for tip in active:
                skeleton.nodes[tip["node"]].on_bed = True
            active = []
            if ci >= len(remaining):
                break

    for tip in active:  # 반복 상한에 걸린 경우(정상 경로에서는 일어나지 않는다)
        land(tip)
    return skeleton


def _bead_layer_offsets(radius: float, pitch: float, odd: bool,
                        shell: Optional[float] = None):
    """축을 중심으로 반경 ``radius`` 원 안에 들어가는 육각 최밀 격자 오프셋.

    홀수 층은 (pitch/2, pitch*sqrt3/6) 만큼 밀어서 아래층 세 구슬이 만드는
    오목한 자리(hollow)에 앉힌다 — 격자 방식과 같은 최밀충전 원리다.

    ``shell`` 을 주면 바깥 테두리 두께만큼만 채운 관(tube)을 만든다. 휨에
    버티는 힘은 대부분 바깥 둘레에서 나오므로, 굵은 트렁크 속까지 꽉 채우는
    것은 구슬 낭비다(실측: 노즐 1mm 거치대에서 속을 채우면 구슬 29만 개).
    """
    if radius < 0.5 * pitch:
        return [(0.0, 0.0)]
    # 최대 굵기에 도달한 줄기와 같은 굵기의 뿌리는 동일 원판을 반복한다.
    # 반경을 반올림하지 않고 정확히 같은 입력만 재사용한다. 호출자가
    # 반환 목록을 바꿔도 다른 층에 영향을 주지 않도록 캐시에는 튜플을 둔다.
    cache = getattr(_bead_layer_offsets, "_offset_cache", None)
    if cache is None:
        cache = {}
        _bead_layer_offsets._offset_cache = cache
    cache_key = (radius, pitch, bool(odd), shell)
    cached = cache.pop(cache_key, None)
    if cached is not None:
        cache[cache_key] = cached
        return list(cached)
    ox, oy = (0.5 * pitch, pitch * math.sqrt(3.0) / 6.0) if odd else (0.0, 0.0)
    row_h = pitch * math.sqrt(3.0) / 2.0
    n = int(math.ceil(radius / row_h)) + 1
    out = []
    for j in range(-n, n + 1):
        y = j * row_h + oy
        shift = 0.5 * pitch if (j % 2) else 0.0
        m = int(math.ceil(radius / pitch)) + 1
        for i in range(-m, m + 1):
            x = i * pitch + shift + ox
            rr = x * x + y * y
            if rr <= radius * radius + 1e-9 and (
                    shell is None or rr >= max(0.0, radius - shell) ** 2 - 1e-9):
                out.append((x, y))
    if not out:
        out = [(0.0, 0.0)]
    if len(cache) >= 256:
        cache.pop(next(iter(cache)))
    cache[cache_key] = tuple(out)
    return out


def skeleton_to_bead_seeds(
    skeleton: SupportSkeleton,
    contact_diameter_mm: float,
    body_diameter_mm: float,
    model_slices: Optional[Sequence] = None,
    heights: Optional[Sequence[float]] = None,
    xy_clearance: float = 0.0,
    max_branch_angle_deg: float = 25.0,
    overlap: float = 0.10,
    include_on_model: bool = True,
    shell_rings: Optional[int] = 2,
) -> List[Tuple[float, float, float, float]]:
    """골격을 구슬로 바꾼다. 반환: (x, y, z, 지름) 목록.

    예전 구현은 엣지마다 구슬 1줄(사슬)만 놓았다. 트렁크가 높이 96mm 에
    굵기 구슬 1개(3.6mm)인 외줄이 되어 옆 지지가 전혀 없었고, 이 프로젝트의
    핵심인 최밀충전(이웃 12개)이 트리 모드에서만 빠져 있었다.

    이제 노드의 ``radius`` 를 **트렁크 단면 반경**으로 보고, 공통 높이
    단계마다 그 원 안을 몸통 구슬로 육각 최밀충전한 원판을 쌓는다.
    원판 층은 ABAB 로 엇갈려 위층 구슬이 아래층 오목한 자리에 앉는다.

    - 층 간격은 '가지가 최대 각도로 기울어도 위아래 구슬이 겹치도록'
      ``min(0.8165*pitch, pitch*cos(각도))`` 로 잡는다(끊김 방지).
    - 모델을 파고드는 구슬: 축(가운데) 구슬은 성장 단계에서 이미 회피가
      보장되므로 조금 밀어내고, 원판 가장자리 구슬은 **밀지 않고 뺀다**.
      하나씩 밀어내면 사슬이 끊긴다(실측: 연결 덩어리 10 -> 38).
    """
    seeds: List[Tuple[float, float, float, float]] = []
    if not skeleton.nodes:
        return seeds
    d_body = body_diameter_mm
    pitch = d_body * (1.0 - overlap)
    # 굵은 트렁크는 바깥 몇 겹만 채운 관으로 만든다(None 이면 속까지 채움).
    shell_mm = None if shell_rings is None else (shell_rings + 0.2) * pitch
    dz_layer = min(pitch * math.sqrt(2.0 / 3.0),
                   pitch * math.cos(math.radians(max_branch_angle_deg)))
    z_base = min(n.z for n in skeleton.nodes)

    axis: List[Tuple[float, float, float, float]] = []
    rim: List[Tuple[float, float, float, float]] = []

    skip = set()
    if not include_on_model:
        # '베드 위에만' 옵션: 모델 윗면에 내려앉은 가지는 통째로 뺀다.
        for idx in range(len(skeleton.nodes)):
            r = idx
            while skeleton.nodes[r].parent is not None:
                r = skeleton.nodes[r].parent
            if skeleton.nodes[r].on_model:
                skip.add(idx)

    for idx, node in enumerate(skeleton.nodes):
        if idx in skip:
            continue
        if node.kind == "contact":
            axis.append((node.x, node.y, node.z, contact_diameter_mm))
        elif node.parent is None:
            axis.append((node.x, node.y, node.z, d_body))
        if node.parent is None:
            if getattr(node, "on_bed", False):
                # 뿌리: 베드 위에 한 층짜리 원판(발)을 깐다
                for ox, oy in _bead_layer_offsets(
                        max(node.radius, 0.5 * d_body) - 0.5 * d_body, pitch, False):
                    if ox or oy:
                        rim.append((node.x + ox, node.y + oy, node.z, d_body))
            continue
        par = skeleton.nodes[node.parent]
        z_hi, z_lo = node.z, par.z
        if z_hi - z_lo < 1e-9:
            continue
        k0 = int(math.ceil((z_lo - z_base) / dz_layer - 1e-9))
        k1 = int(math.floor((z_hi - z_base) / dz_layer + 1e-9))
        for k in range(k0, k1 + 1):
            zz = z_base + k * dz_layer
            t = (z_hi - zz) / (z_hi - z_lo)  # 0=자식(위) 1=부모(아래)
            cx = node.x + (par.x - node.x) * t
            cy = node.y + (par.y - node.y) * t
            rad = node.radius + (par.radius - node.radius) * t
            d = contact_diameter_mm if (node.kind == "contact" and t < 0.5) else d_body
            axis.append((cx, cy, zz, d))
            inner = max(0.0, rad - 0.5 * d_body)
            if inner >= 0.5 * pitch:
                for ox, oy in _bead_layer_offsets(inner, pitch, bool(k % 2),
                                                  shell=shell_mm):
                    if ox or oy:
                        rim.append((cx + ox, cy + oy, zz, d_body))

    if model_slices is not None and heights is not None:
        cache: dict = {}
        axis = [_nudge_out_of_model(x, y, z, dd, model_slices, heights,
                                    xy_clearance, _cache=cache)
                for x, y, z, dd in axis]
        rim = [s for s in rim
               if _nudge_out_of_model(*s, model_slices, heights, xy_clearance,
                                      _cache=cache)[:2] == s[:2]]
    seeds = _dedupe(axis, rim, 0.5 * pitch)
    return seeds


def _dedupe(first, second, min_sep):
    """겹쳐 놓인 구슬을 하나만 남긴다.

    가지가 합쳐지는 곳에서는 두 원판이 겹치므로 같은 자리에 구슬이 두 번
    놓인다. 겹쳐 놓인 구슬은 압출량만 두 배로 쓰고 모양은 그대로다.

    단, **축(사슬) 구슬은 거의 같은 자리일 때만** 지운다. 합쳐지기 직전의
    두 가지는 서로 가깝게 나란히 내려오는데, 넉넉한 기준으로 한쪽 사슬
    구슬을 지우면 그 가지가 이웃 가지 구슬에 겨우 스치듯 붙어 끊긴다
    (실측: 캔틸레버 모델에서 가지 2개, 구슬 46개가 떨어져 나갔다).
    원판 가장자리(rim) 구슬만 넉넉한 기준을 쓴다.
    """
    first, second = list(first), list(second)
    if len(first) + len(second) < 2:
        return first + second
    from scipy.spatial import cKDTree
    keep_first = [True] * len(first)
    if len(first) > 1:
        P = np.array([(x, y, z) for x, y, z, _ in first])
        for i, j in sorted(cKDTree(P).query_pairs(0.3 * min_sep)):
            if keep_first[i] and keep_first[j]:
                keep_first[j] = False
    kept = [b for b, k in zip(first, keep_first) if k]
    if not second:
        return kept
    Q = np.array([(x, y, z) for x, y, z, _ in second])
    if kept:
        dist, _ = cKDTree(np.array([(x, y, z) for x, y, z, _ in kept])).query(Q)
        cand = [i for i in range(len(second)) if dist[i] >= min_sep]
    else:
        cand = list(range(len(second)))
    keep_second = {i: True for i in cand}
    if len(cand) > 1:
        sub = Q[cand]
        for a, b in sorted(cKDTree(sub).query_pairs(min_sep)):
            ia, ib = cand[a], cand[b]
            if keep_second[ia] and keep_second[ib]:
                keep_second[ib] = False
    return kept + [second[i] for i in cand if keep_second[i]]


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
    from bisect import bisect_left

    def nearest_height(target):
        # 슬라이싱 높이는 오름차순이다. 동률이면 기존 min과 동일하게
        # 앞쪽 인덱스를 선택하며, 중복 높이에서도 첫 번째를 돌려준다.
        if len(heights) == 0:
            raise ValueError("min() arg is an empty sequence")
        right = bisect_left(heights, target)
        if right == 0:
            return 0
        if right == len(heights):
            return bisect_left(heights, heights[-1])
        chosen = right - 1 if abs(heights[right - 1] - target) <= abs(heights[right] - target) else right
        return bisect_left(heights, heights[chosen])

    r = 0.5 * d
    lo = nearest_height(z - r)
    hi = nearest_height(z + r)
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
    padding = xy_clearance + r + 0.15
    buffer_key = ("keep_out", lo, hi, padding)
    if _cache is not None and buffer_key in _cache:
        keep_out = _cache[buffer_key]
    else:
        keep_out = model.buffer(padding)
        if _cache is not None:
            _cache[buffer_key] = keep_out
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
    slenderness: float = 8.0,
) -> None:
    """노드마다 **트렁크 단면 반경**을 매긴다(요구사항 10 + 좌우 흔들림).

    두 조건 중 큰 쪽을 쓴다.

    1. 짐: 떠받치는 접촉점 수의 제곱근에 비례(단면적 ∝ 짐).
    2. 세장비: 이 노드 위로 서 있는 높이 / ``slenderness``. 위에서 아래로
       갈수록 굵어지는 원추형이 된다. 휨 모멘트가 가장 큰 곳이 밑동이라
       밑동이 가장 굵어야 한다 — 예전 구현은 높이 96mm 트렁크의 밑동이
       구슬 1개(3.6mm, 세장비 26)였다.

    ``radius`` 는 이제 구슬 크기가 아니라 '이 높이에서 트렁크 원판의 반경'
    이다. 구슬은 항상 몸통 지름이고, 굵기는 원판 안의 구슬 개수로 낸다.
    """
    n = len(skeleton.nodes)
    if n == 0:
        return
    if max_trunk_diameter_mm is None:
        max_trunk_diameter_mm = body_diameter_mm * 15.0

    load = [0] * n
    ztop = [skeleton.nodes[i].z for i in range(n)]
    order: List[int] = []
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
            stack.extend(skeleton.children[i])
        order.extend(reversed(seq))  # 리프부터

    for r in skeleton.roots():
        walk(r)
    for i in range(n):
        if not visited[i]:
            walk(i)

    for i in order:
        node = skeleton.nodes[i]
        own = 1 if node.kind == "contact" else 0
        load[i] = own + sum(load[c] for c in skeleton.children[i])
        for c in skeleton.children[i]:
            ztop[i] = max(ztop[i], ztop[c])

    for i, node in enumerate(skeleton.nodes):
        if node.kind == "contact":
            node.radius = 0.5 * contact_diameter_mm
            continue
        d_load = body_diameter_mm * math.sqrt(max(1, load[i])) * 0.5
        d_slender = (ztop[i] - node.z) / max(slenderness, 1e-6)
        d = max(body_diameter_mm, d_load, d_slender)
        node.radius = 0.5 * min(d, max_trunk_diameter_mm)


def add_bracing(
    skeleton: SupportSkeleton,
    body_diameter_mm: float,
    model_slices: Optional[Sequence] = None,
    heights: Optional[Sequence[float]] = None,
    xy_clearance: float = 0.0,
    max_distance_mm: Optional[float] = None,
    brace_angle_deg: float = 45.0,
    min_height_mm: Optional[float] = None,
    overlap: float = 0.10,
) -> Tuple[List[Tuple[float, float, float, float]], int, int]:
    """가까운 트렁크끼리 서로 붙잡게 한다. 반환: (구슬, 베드연결수, 가새수).

    트렁크가 각자 혼자 서 있으면 아무리 굵어도 흔들림을 혼자 받는다.
    이웃 트렁크를 최소 신장 트리(MST)로 골라 두 가지로 잇는다.

    - **베드 연결**: 밑동끼리 베드 위에서 잇는 구슬 줄. 베드에 얹혀 있어
      출력에 문제가 없다.
    - **X 가새**: 한쪽 트렁크의 높은 곳에서 반대쪽 트렁크의 낮은 곳으로
      ``brace_angle_deg``(수직 기준) 로 비스듬히 내려가는 구슬 줄을 양방향으로.
      수평 가로대는 가운데가 허공에 뜨므로 쓰지 않는다 — 45° 이하 경사면은
      아래 구슬 위에 차례로 얹히며 출력된다.

    모델을 지나가는 연결은 건너뛴다.
    """
    seeds: List[Tuple[float, float, float, float]] = []
    roots = [i for i in skeleton.roots()
             if getattr(skeleton.nodes[i], "on_bed", False)]
    if len(roots) < 2:
        return seeds, 0, 0
    if max_distance_mm is None:
        max_distance_mm = body_diameter_mm * 20.0
    if min_height_mm is None:
        min_height_mm = body_diameter_mm * 6.0
    pitch = body_diameter_mm * (1.0 - overlap)

    # 각 루트에서 '가장 무거운 자식'을 따라 올라가는 주축 경로
    def main_path(root: int):
        path = [root]
        cur = root
        while skeleton.children[cur]:
            cur = max(skeleton.children[cur],
                      key=lambda c: (skeleton.nodes[c].radius, -skeleton.nodes[c].z))
            path.append(cur)
        return path  # 아래 -> 위

    paths = {r: main_path(r) for r in roots}

    def axis_at(root: int, zz: float):
        p = paths[root]
        nodes = skeleton.nodes
        if zz < nodes[p[0]].z or zz > nodes[p[-1]].z:
            return None
        for a, b in zip(p, p[1:]):
            za, zb = nodes[a].z, nodes[b].z
            if za <= zz <= zb:
                t = 0.0 if zb - za < 1e-9 else (zz - za) / (zb - za)
                return (nodes[a].x + (nodes[b].x - nodes[a].x) * t,
                        nodes[a].y + (nodes[b].y - nodes[a].y) * t,
                        nodes[a].radius + (nodes[b].radius - nodes[a].radius) * t)
        n = nodes[p[-1]]
        return (n.x, n.y, n.radius)

    # 후보 연결: 각 트렁크에서 가까운 이웃 몇 개. 모델에 막히는 연결은 아래
    # 루프에서 건너뛰고 다음 후보를 쓰도록, 크루스칼 방식으로 고른다
    # (MST 를 먼저 정해 버리면 막힌 간선 하나 때문에 양쪽이 영영 안 이어진다).
    from scipy.spatial import cKDTree

    P = np.array([(skeleton.nodes[r].x, skeleton.nodes[r].y) for r in roots])
    m = len(roots)
    kq = min(m, 7)
    dists, nbrs = cKDTree(P).query(P, k=kq)
    cand = set()
    for a in range(m):
        for dd, b in zip(np.atleast_1d(dists[a]), np.atleast_1d(nbrs[a])):
            if b != a and dd <= max_distance_mm:
                cand.add((min(a, int(b)), max(a, int(b)), float(dd)))
    cand = sorted(cand, key=lambda t: t[2])
    uf = list(range(m))

    def find(u):
        while uf[u] != u:
            uf[u] = uf[uf[u]]
            u = uf[u]
        return u

    cache: dict = {}

    def clear(x, y, z, d):
        if model_slices is None or heights is None:
            return True
        return _nudge_out_of_model(x, y, z, d, model_slices, heights,
                                   xy_clearance, _cache=cache)[:2] == (x, y)

    def chain(p0, p1, d):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        k = max(1, int(math.ceil(L / pitch)))
        pts = [tuple(p0 + (p1 - p0) * (i / k)) + (d,) for i in range(k + 1)]
        if not all(clear(*q) for q in pts):
            return None
        return pts

    n_ties = n_braces = 0
    tan_b = math.tan(math.radians(brace_angle_deg))
    for a, b, dist in cand:
        if find(a) == find(b):
            continue
        ra, rb = roots[a], roots[b]
        na, nb = skeleton.nodes[ra], skeleton.nodes[rb]
        # 1) 베드 연결: 서로의 원판 가장자리 사이만 잇는다
        gap = dist - na.radius - nb.radius
        if gap > 0:
            ux, uy = (nb.x - na.x) / dist, (nb.y - na.y) / dist
            z_b = min(na.z, nb.z)
            q0 = (na.x + ux * na.radius, na.y + uy * na.radius, z_b)
            q1 = (nb.x - ux * nb.radius, nb.y - uy * nb.radius, z_b)
            pts = chain(q0, q1, body_diameter_mm)
            if not pts:
                continue  # 모델에 막힘 -> 다음 후보로
            seeds.extend(pts)
            n_ties += 1
        uf[find(a)] = find(b)
        # 2) X 가새: 두 트렁크가 모두 충분히 높을 때만
        top_a = skeleton.nodes[paths[ra][-1]].z
        top_b = skeleton.nodes[paths[rb][-1]].z
        z_top = min(top_a, top_b)
        if z_top - max(na.z, nb.z) < min_height_mm:
            continue
        zz = max(na.z, nb.z) + body_diameter_mm
        while True:
            pa = axis_at(ra, zz)
            pb = axis_at(rb, zz)
            if pa is None or pb is None:
                break
            span = math.hypot(pa[0] - pb[0], pa[1] - pb[1]) - pa[2] - pb[2]
            if span <= pitch:
                break  # 이미 원판끼리 붙어 있다
            rise = span / tan_b
            if zz + rise > z_top - body_diameter_mm:
                break
            placed = False
            for (s, e) in ((ra, rb), (rb, ra)):
                # 위쪽 끝의 트렁크 반경은 높이에 따라 달라지므로(원추형),
                # 실제 두 끝점 사이 수평거리로 기울기를 몇 번 다시 맞춘다.
                r_s = rise
                q0 = q1 = None
                for _ in range(4):
                    ps, pe = axis_at(s, zz), axis_at(e, zz + r_s)
                    if ps is None or pe is None:
                        q0 = None
                        break
                    ux, uy = pe[0] - ps[0], pe[1] - ps[1]
                    ll = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / ll, uy / ll
                    q0 = (ps[0] + ux * ps[2], ps[1] + uy * ps[2], zz)
                    q1 = (pe[0] - ux * pe[2], pe[1] - uy * pe[2], zz + r_s)
                    r_s = max(r_s, math.hypot(q1[0] - q0[0], q1[1] - q0[1]) / tan_b)
                if q0 is None or q1[2] > z_top - body_diameter_mm:
                    continue
                if math.hypot(q1[0] - q0[0], q1[1] - q0[1]) > \
                        (q1[2] - q0[2]) * tan_b * 1.05:
                    continue
                pts = chain(q0, q1, body_diameter_mm)
                if pts:
                    seeds.extend(pts)
                    placed = True
            if placed:
                n_braces += 1
            zz += rise
    return seeds, n_ties, n_braces


def structure_report(seeds, bed_z: float = 0.0, touch: float = 0.98):
    """구슬 더미를 '구조물'로 점검한다: 실제로 겹쳐 붙은 구슬끼리만 연결로
    보고, 베드에 이어지지 않은 덩어리와 아래 받침 없이 허공에 찍히는 구슬을
    센다. 설계값이 아니라 최종 좌표에서 잰다.
    """
    if not seeds:
        return {"beads": 0, "components": 0, "floating_beads": 0,
                "unsupported_beads": 0}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    A = np.asarray(seeds, dtype=float)
    X, R = A[:, :3], 0.5 * A[:, 3]
    n = len(X)
    pairs = cKDTree(X).query_pairs(2.0 * R.max(), output_type="ndarray")
    if len(pairs):
        dist = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        pairs = pairs[dist < (R[pairs[:, 0]] + R[pairs[:, 1]]) * touch]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs)
                   else ([], ([], [])), shape=(n, n))
    n_comp, lab = connected_components(g, directed=False)
    on_bed = (X[:, 2] - R) <= bed_z + 0.05
    grounded = np.zeros(n_comp, dtype=bool)
    grounded[np.unique(lab[on_bed])] = True
    floating = int((~grounded[lab]).sum())

    sup = on_bed.copy()
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            dz = X[a, 2] - X[b, 2]
            if dz > 0.15 * R[a] and math.hypot(*(X[a, :2] - X[b, :2])) <= dz:
                sup[a] = True
    return {"beads": n, "components": int(n_comp), "floating_beads": floating,
            "unsupported_beads": int((~sup).sum())}


def fix_residual_collisions(
    seeds: List[Tuple[float, float, float, float]],
    mesh,
    xy_clearance: float,
    max_iters: int = 3,
    simplify_above_faces: int = 300_000,
    chunk: int = 400,
    z_gap: Optional[float] = None,
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
        surface_normals = np.empty_like(P)
        inside = np.zeros(len(P), dtype=bool)
        for s in range(0, len(P), chunk):
            e = min(s + chunk, len(P))
            c_, d_, _t = pq.on_surface(P[s:e])
            closest[s:e] = c_
            dist[s:e] = d_
            surface_normals[s:e] = probe.face_normals[_t]
            # on_surface 는 '부호 없는' 거리를 준다. 모델 **안쪽** 깊숙이
            # 있는 구슬은 표면까지 멀어서 오히려 안전해 보인다(실측: 상자
            # 한가운데 구슬이 표면까지 10mm 라 통과해 버렸다).
            # 안/밖을 따로 판정해야 한다.
            inside[s:e] = pq.signed_distance(P[s:e]) > 0
        need_r = 0.5 * D + xy_clearance
        # 모델 윗면에 실제로 앉은 구슬은 XY 여유만큼 위로 띄우면 안 된다.
        # 반지름과의 거리 차이가 수치 오차 수준인 접촉만 인정하고, 위를
        # 향한 면의 법선과 실제 접촉 방향이 일치해야 한다. 모서리 옆이나
        # 벽 근처에 떠 있는 구슬은 최근접 면이 위를 향해도 착지가 아니다.
        outward = P - closest
        alignment = np.einsum("ij,ij->i", outward, surface_normals) / np.maximum(dist, 1e-9)
        contact_tolerance = np.minimum(1e-3, 0.01 * D)
        on_model_surface = (
            ~inside
            & (np.abs(dist - 0.5 * D) <= contact_tolerance)
            & (outward[:, 2] > 1e-8)
            & (surface_normals[:, 2] > 1e-8)
            & (alignment >= 1.0 - 1e-4)
        )
        need_r = np.where(on_model_surface, 0.5 * D, need_r)
        if z_gap is not None:
            # 바로 위에서 내려다보는 오버행 면과의 간격은 XY 여유가 아니라
            # z 간격(contact_z_gap)으로 본다. 모든 방향을 XY 여유로 재면
            # 접촉 구슬이 전부 '관통'으로 잡혀 아래로 밀리거나 빠진다
            # (실측: 테이블 모델 접촉 구슬 168개가 전부 빠졌다).
            vec = closest - P
            above = (vec[:, 2] > 0) & (vec[:, 2] >= 0.7 * np.maximum(dist, 1e-9))
            need_r = np.where(above, 0.5 * D + min(z_gap, xy_clearance), need_r)
        bad = (dist < need_r - 1e-3) | inside
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



def settle_collisions(seeds, mesh, xy_clearance: float,
                      max_push_ratio: float = 0.35,
                      z_gap: Optional[float] = None,
                      model_slices: Optional[Sequence] = None,
                      heights: Optional[Sequence[float]] = None):
    """3D 메쉬로 마지막 관통 검사. 조금만 밀면 되는 구슬은 밀고, 멀리
    밀어야 하는 구슬은 **뺀다**. 반환: (구슬, 뺀 개수).

    멀리 밀린 구슬은 이웃과 떨어져 혼자 공중에 뜬다(실측: 개별 밀어내기로
    연결 덩어리가 10 -> 38 로 쪼개졌다). 그런 구슬은 있어도 서포터 역할을
    못 하고 출력 불량만 만든다.
    """
    if not seeds or mesh is None:
        return seeds, 0
    # 3D 거리 계산은 비싸다(구슬 5천 개, 삼각형 수십만 개에서 1~2분).
    # 슬라이스 단면에서 충분히 먼 구슬은 확실히 안전하므로 뺀다.
    check = list(range(len(seeds)))
    if model_slices is not None and heights is not None and len(heights) > 1:
        import shapely
        from shapely.ops import unary_union
        det_h = heights[1] - heights[0]
        near = []
        cache: dict = {}
        for i, (x, y, z, d) in enumerate(seeds):
            r = 0.5 * d
            lo = max(0, int(math.floor((z - r - det_h - heights[0]) / det_h)))
            hi = min(len(heights) - 1,
                     int(math.ceil((z + r + det_h - heights[0]) / det_h)))
            key = (lo, hi, round(r, 3))
            if key not in cache:
                parts = [clean(model_slices[j]) for j in range(lo, hi + 1)]
                parts = [g for g in parts if not g.is_empty]
                if parts:
                    g = unary_union(parts).buffer(r + xy_clearance + det_h,
                                                  quad_segs=4)
                    shapely.prepare(g)
                else:
                    g = None
                cache[key] = g
            g = cache[key]
            if g is not None and shapely.contains_xy(g, x, y):
                near.append(i)
        check = near
    if not check:
        return list(seeds), 0
    sub = [seeds[i] for i in check]
    pushed_sub = fix_residual_collisions(sub, mesh, xy_clearance, z_gap=z_gap)
    pushed = list(seeds)
    for i, q in zip(check, pushed_sub):
        pushed[i] = q
    out = []
    dropped = 0
    for (x0, y0, z0, d), (x1, y1, z1, _d) in zip(seeds, pushed):
        # 옆으로 밀리면 이웃과 떨어지지만, 아래로 조금 밀리는 것(오버행 바로
        # 밑 접촉 구슬)은 사슬 방향이라 연결이 유지된다.
        if math.hypot(x1 - x0, y1 - y0) > max_push_ratio * d or \
                abs(z1 - z0) > 0.6 * d:
            dropped += 1
            continue
        out.append((x1, y1, z1, d))
    return out, dropped


def _touch_components(X, R, touch: float = 0.98):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(X)
    pairs = cKDTree(X).query_pairs(2.0 * R.max(), output_type="ndarray")
    if len(pairs):
        dist = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        pairs = pairs[dist < (R[pairs[:, 0]] + R[pairs[:, 1]]) * touch]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs)
                   else ([], ([], [])), shape=(n, n))
    return connected_components(g, directed=False)


def prune_floating(seeds, mesh, bed_z: float, xy_clearance: float):
    """실제 받침에서 아래→위 순서로 연결되지 않는 구슬을 뺀다.

    XY 여유는 바닥의 공중 간격이 아니다. 벽에 가깝거나 위쪽으로만
    연결된 구슬을 모델 위에 착지한 것으로 인정하지 않는다.
    ``xy_clearance`` 인자는 기존 호출 호환을 위해 유지한다.
    """
    if not seeds:
        return seeds, 0
    from .printability import supported_mask
    keep = supported_mask(seeds, mesh, bed_z)
    return [s for s, k in zip(seeds, keep) if k], int((~keep).sum())
