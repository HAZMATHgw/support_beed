# -*- coding: utf-8 -*-
"""서포터 구조 안정성 시뮬레이션.

프린터 플레이트가 좌우로 흔들릴 때 서포터가 버티는지, 구슬이 목(neck)에서
떨어져 나가지 않는지를 검사한다.

물리 모델
---------
구슬 하나하나를 절점으로, 서로 눌린 접촉면(neck)을 부재로 보는 격자 구조로
다룬다. 플레이트가 가속도 ``a`` 로 흔들리면 높이 z 위에 있는 모든 구슬의
질량에 관성력이 걸리고, 그 힘은 z 를 지나는 모든 neck 의 단면적이 나눠서
받는다. 따라서 임의의 높이에서

    전단응력 = (그 위 질량) x 가속도 / (그 높이를 지나는 neck 단면적 합)

이 값이 재료의 층간 접합 전단강도를 넘으면 그 높이에서 서포터가 끊어진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .params import SupportBeadParams, SupportGenParams

#: PLA 층간 접합 전단강도(MPa = N/mm^2). 벌크 강도가 아니라 층 사이 접합
#: 강도이며, 문헌값 20~30 MPa 중 보수적으로 잡았다.
DEFAULT_SHEAR_STRENGTH_MPA = 15.0

#: 재료 밀도 (g/mm^3). PLA ~1.24 g/cm^3.
DEFAULT_DENSITY_G_MM3 = 1.24e-3


@dataclass
class StabilityReport:
    n_beads: int
    n_components: int
    largest_component_ratio: float
    n_unsupported: int          # 아래에 아무것도 없는 구슬
    n_weakly_attached: int      # 이웃이 3개 이하
    worst_height_mm: float      # 가장 취약한 높이
    worst_shear_mpa: float      # 그 높이의 전단응력
    safety_factor: float        # 강도 / 응력. 1 미만이면 끊어진다
    tipping_safety_factor: float
    footprint_mm2: float

    def summary(self) -> str:
        ok = "OK" if self.safety_factor >= 1.0 else "위험"
        tip = "OK" if self.tipping_safety_factor >= 1.0 else "위험"
        return (
            f"구슬 {self.n_beads:,}개 / 덩어리 {self.n_components}개 "
            f"(최대 덩어리 {self.largest_component_ratio*100:.1f}%)\n"
            f"  받쳐지지 않는 구슬 {self.n_unsupported:,}개, "
            f"약하게 붙은 구슬 {self.n_weakly_attached:,}개\n"
            f"  최약 높이 z={self.worst_height_mm:.1f}mm 에서 "
            f"전단응력 {self.worst_shear_mpa:.3f} MPa "
            f"-> 안전계수 {self.safety_factor:.2f} [{ok}]\n"
            f"  전도 안전계수 {self.tipping_safety_factor:.2f} [{tip}], "
            f"바닥 면적 {self.footprint_mm2:.0f} mm^2"
        )


def bead_positions(plan, gen: SupportGenParams) -> Tuple[np.ndarray, np.ndarray]:
    """(위치 Nx3, 지름 N) 배열."""
    pts: List[Tuple[float, float, float]] = []
    diam: List[float] = []
    for layer in plan.layers:
        zc = layer["z_bottom"] + 0.5 * gen.layer_height_mm
        for bead in layer["beads"]:
            pts.append((bead["x"], bead["y"], zc))
            diam.append(bead["d"])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.asarray(pts, dtype=float), np.asarray(diam, dtype=float)


def contact_pairs(points: np.ndarray, pitch: float, tol: float = 1.02):
    """서로 눌려 붙은 구슬 쌍."""
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return np.zeros((0, 2), dtype=int)
    tree = cKDTree(points)
    return tree.query_pairs(pitch * tol, output_type="ndarray")


def analyze(plan, gen: SupportGenParams, contact: SupportBeadParams,
            mesh=None,
            lateral_g: float = 0.5,
            shear_strength_mpa: float = DEFAULT_SHEAR_STRENGTH_MPA,
            density_g_mm3: float = DEFAULT_DENSITY_G_MM3) -> Optional[StabilityReport]:
    """흔들림에 대한 안정성을 평가한다.

    ``lateral_g`` 는 플레이트가 흔들릴 때 걸리는 수평 가속도를 중력 대비
    배수로 준다. 0.5 는 "중력의 절반만큼 옆으로 흔들린다"는 뜻으로,
    급가속하는 벨트 구동 프린터에서 흔히 나오는 수준이다.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    P, D = bead_positions(plan, gen)
    n = len(P)
    if n == 0:
        return None

    pitch = contact.pitch_mm()
    pairs = contact_pairs(P, pitch)

    # --- 연결성 -----------------------------------------------------------
    if len(pairs):
        adj = sp.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        adj = adj + adj.T
        n_comp, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels)
        largest = sizes.max() / n
        degree = np.bincount(pairs.ravel(), minlength=n)
    else:
        n_comp, largest = n, 1.0 / n
        degree = np.zeros(n, dtype=int)

    # --- 아래에서 받쳐지는가 ----------------------------------------------
    n_unsupported = int(count_unsupported(P, pitch, gen, mesh))

    # --- 흔들림 전단 ------------------------------------------------------
    r = 0.5 * D
    bead_mass_g = density_g_mm3 * (4.0 / 3.0) * math.pi * r ** 3
    a = lateral_g * 9806.65  # mm/s^2
    # 힘(N) = 질량(kg) * 가속도(m/s^2) = (g/1000) * (mm/s^2 /1000)
    force_per_bead_n = (bead_mass_g / 1000.0) * (a / 1000.0)

    neck_r = contact.contact_disc_radius_mm()
    neck_area = math.pi * neck_r * neck_r  # mm^2

    worst_stress = 0.0
    worst_z = 0.0
    if neck_area > 0 and len(pairs):
        zs = P[:, 2]
        # 각 층 경계마다: 위쪽 질량이 만드는 수평력 / 그 경계를 지나는 neck 수
        levels = np.unique(np.round(zs, 4))
        z_lo = P[pairs[:, 0], 2]
        z_hi = P[pairs[:, 1], 2]
        pair_lo = np.minimum(z_lo, z_hi)
        pair_hi = np.maximum(z_lo, z_hi)
        for z in levels[:-1]:
            above = zs > z + 1e-6
            if not above.any():
                continue
            shear_force = force_per_bead_n[above].sum()
            crossing = ((pair_lo <= z + 1e-6) & (pair_hi > z + 1e-6)).sum()
            if crossing == 0:
                # 이 높이를 지나는 접촉이 하나도 없다 = 완전히 끊어져 있다
                worst_stress = float("inf")
                worst_z = float(z)
                break
            stress = shear_force / (crossing * neck_area)
            if stress > worst_stress:
                worst_stress, worst_z = stress, float(z)

    safety = (shear_strength_mpa / worst_stress) if worst_stress > 0 else float("inf")

    # --- 전도(넘어짐) -----------------------------------------------------
    z0 = P[:, 2].min()
    base = P[P[:, 2] < z0 + pitch]
    if len(base) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(base[:, :2])
            footprint = float(hull.volume)  # 2D 에서 volume = 면적
            half_width = math.sqrt(footprint / math.pi)
        except Exception:
            footprint = 0.0
            half_width = float(np.ptp(base[:, 0]) + np.ptp(base[:, 1])) / 4
    else:
        footprint = 0.0
        half_width = max(pitch, 1e-6)

    com_z = float(np.average(P[:, 2], weights=bead_mass_g))
    # 복원 모멘트(무게 x 폭) 대 전도 모멘트(수평력 x 무게중심 높이)
    tipping = (half_width / (lateral_g * com_z)) if com_z > 0 else float("inf")

    return StabilityReport(
        n_beads=n,
        n_components=int(n_comp),
        largest_component_ratio=float(largest),
        n_unsupported=n_unsupported,
        n_weakly_attached=int((degree <= 3).sum()),
        worst_height_mm=worst_z,
        worst_shear_mpa=float(worst_stress),
        safety_factor=float(safety),
        tipping_safety_factor=float(tipping),
        footprint_mm2=footprint,
    )


def count_unsupported(points: np.ndarray, pitch: float,
                      gen: SupportGenParams, mesh=None) -> int:
    """아래에 아무것도 없는(공중에 뜬) 구슬 수."""
    if len(points) == 0:
        return 0
    from scipy.spatial import cKDTree

    z = points[:, 2]
    bed_level = z.min() + 1e-6
    on_bed = z <= bed_level

    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, pitch * 1.02)
    has_below = np.zeros(len(points), dtype=bool)
    for i, idx in enumerate(neighbors):
        for j in idx:
            if j != i and points[j, 2] < z[i] - 1e-6:
                has_below[i] = True
                break

    on_model = np.zeros(len(points), dtype=bool)
    if mesh is not None:
        try:
            origins = points.copy()
            origins[:, 2] -= 1e-3
            dirs = np.tile([0.0, 0.0, -1.0], (len(points), 1))
            on_model = mesh.ray.intersects_any(origins, dirs)
        except Exception:
            pass

    return int((~(on_bed | has_below | on_model)).sum())


def prune_unsupported_beads(plan, gen: SupportGenParams,
                            contact: SupportBeadParams, mesh=None) -> int:
    """아래에 받쳐줄 것이 없는 구슬을 아래층부터 훑어 제거한다.

    공중에 뜬 구슬은 실제로 인쇄되지 않고 노즐에 끌려다니며 출력을 망친다.
    아래층부터 올라가며 판정하므로, 어떤 구슬을 지우면 그 위에 얹혀 있던
    구슬도 같은 패스에서 함께 지워진다.

    반환값은 제거한 구슬 수.
    """
    from scipy.spatial import cKDTree

    pitch = contact.pitch_mm()
    removed = 0
    kept_pts: List[Tuple[float, float, float]] = []
    tree = None
    dirty = True

    for layer in plan.layers:
        if not layer["beads"]:
            continue
        zc = layer["z_bottom"] + 0.5 * gen.layer_height_mm
        # 베드에 닿는 구슬은 항상 통과.
        #
        # 예전에는 z_bottom <= 0 으로 봤는데, 첫 구슬 층의 z_bottom 은 정확히
        # 0 이 아니라 반지름 보정 때문에 살짝 떠 있다(예: 0.0622). 그래서 맨
        # 아래층이 통째로 제거되어 바닥 면적이 오히려 줄고 전도에 약해졌다.
        # 구 중심이 반지름 이내면 베드에 닿은 것으로 본다.
        max_radius = 0.5 * max(b["d"] for b in layer["beads"])
        on_bed = zc <= max_radius * 1.05

        if not on_bed:
            if dirty and kept_pts:
                tree = cKDTree(np.asarray(kept_pts))
                dirty = False

            survivors = []
            xy = np.array([[b["x"], b["y"]] for b in layer["beads"]])
            # 모델 위에 얹혀 있는지 한 번에 판정
            on_model = np.zeros(len(xy), dtype=bool)
            if mesh is not None:
                try:
                    origins = np.column_stack([xy, np.full(len(xy), zc - 1e-3)])
                    dirs = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
                    on_model = mesh.ray.intersects_any(origins, dirs)
                except Exception:
                    pass

            for i, bead in enumerate(layer["beads"]):
                supported = bool(on_model[i])
                if not supported and tree is not None:
                    p = np.array([bead["x"], bead["y"], zc])
                    for j in tree.query_ball_point(p, pitch * 1.02):
                        if kept_pts[j][2] < zc - 1e-6:
                            supported = True
                            break
                if supported:
                    survivors.append(bead)
                else:
                    removed += 1
            layer["beads"] = survivors

        for bead in layer["beads"]:
            kept_pts.append((bead["x"], bead["y"], zc))
        dirty = True

    return removed
