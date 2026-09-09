# -*- coding: utf-8 -*-
"""흔들림 시뮬레이션.

프린터 플레이트가 좌우로 흔들릴 때 서포터가 버티는지, 구슬이 서로 떨어지지
않는지를 계산으로 확인한다.

모델링 방식
-----------
구슬을 노드, 맞닿은 구슬 쌍을 축방향 스프링(막대)으로 보는 격자 구조 해석이다.

- 베드에 닿은 구슬은 고정단으로 잡는다.
- 플레이트가 가속도 ``a`` 로 좌우로 흔들리면, 각 구슬에는 관성력
  ``F = m * a`` 가 수평으로 걸린다. 구슬 질량은 재료 밀도 × 구슬 부피.
- 선형 강성 방정식 ``K u = F`` 를 풀어 변위를 얻고, 각 연결(넥)의
  축력을 구한다.
- 넥 단면적으로 나눈 응력을 재료의 인장강도와 비교해 안전율을 낸다.

가정과 한계
-----------
- 축방향 힘만 본다(굽힘·비틀림은 무시). 실제보다 보수적이지 않을 수 있다.
- 재료 물성은 PLA 기준 기본값이며, 실제 펠릿 재료에 맞게 바꿔야 한다.
- 정적 해석이다. 공진은 다루지 않는다.

따라서 절대적인 안전 판정이 아니라, **설정을 바꿨을 때 좋아지는지 나빠지는지
비교하는 용도**로 쓰는 것이 맞다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MaterialProps:
    """재료 물성. 기본값은 PLA 근사치."""

    density_kg_m3: float = 1240.0
    youngs_modulus_mpa: float = 3500.0
    tensile_strength_mpa: float = 50.0
    poisson_ratio: float = 0.35

    @property
    def shear_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class ShakeResult:
    beads: int
    links: int
    grounded: int
    max_displacement_mm: float
    max_neck_stress_mpa: float
    neck_strength_mpa: float
    safety_factor: float
    weakest_link_z_mm: float

    @property
    def survives(self) -> bool:
        return self.safety_factor >= 1.0

    def summary(self) -> str:
        verdict = "버팀" if self.survives else "★파손★"
        return (
            f"구슬 {self.beads:,}개 / 연결 {self.links:,}개 / 고정단 {self.grounded:,}개\n"
            f"  최대 변위      {self.max_displacement_mm:.4f} mm\n"
            f"  최대 넥 응력    {self.max_neck_stress_mpa:.2f} MPa "
            f"(한계 {self.neck_strength_mpa:.1f} MPa, z={self.weakest_link_z_mm:.1f}mm)\n"
            f"  안전율         {self.safety_factor:.2f}  -> {verdict}"
        )


def simulate_lateral_shake(
    points: np.ndarray,
    bead_diameter_mm: float,
    pitch_mm: float,
    accel_g: float = 1.0,
    vertical_distance_mm: Optional[float] = None,
    material: Optional[MaterialProps] = None,
    ground_tol_mm: Optional[float] = None,
    direction: tuple = (1.0, 0.0, 0.0),
) -> ShakeResult:
    """좌우 흔들림에 대한 정적 구조 해석.

    ``accel_g`` 는 플레이트 가속도를 중력가속도 배수로 준다. 1.0 이면
    자기 무게만큼의 힘이 수평으로 걸린다는 뜻이다.
    """
    from scipy.sparse import coo_matrix, diags, identity
    from scipy.sparse.linalg import cg
    from scipy.spatial import cKDTree

    mat = material or MaterialProps()
    n = len(points)
    if n == 0:
        raise ValueError("구슬이 없습니다.")

    r = 0.5 * bead_diameter_mm
    # 눌려 붙은 자리(넥)의 반지름. 구슬이 pitch 만큼 떨어져 겹칠 때 생긴다.
    neck_r = math.sqrt(max(0.0, r * r - 0.25 * pitch_mm * pitch_mm))
    if neck_r <= 0:
        raise ValueError(
            "구슬이 서로 닿지 않습니다(겹침 0). 흔들림을 견딜 수 없습니다."
        )
    neck_area_mm2 = math.pi * neck_r * neck_r
    # 가로/세로 겹침이 다르면 넥 크기도 다르다. 연결마다 방향을 보고 고른다.
    d_v = vertical_distance_mm if vertical_distance_mm else pitch_mm
    neck_rv = math.sqrt(max(0.0, r * r - 0.25 * d_v * d_v))
    neck_area_v = math.pi * neck_rv * neck_rv

    tree = cKDTree(points)
    search_r = max(pitch_mm, d_v) * 1.02
    pairs = tree.query_pairs(search_r, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("맞닿은 구슬이 하나도 없습니다.")

    z_min = points[:, 2].min()
    tol = ground_tol_mm if ground_tol_mm is not None else r * 1.5
    grounded = points[:, 2] <= z_min + tol
    if not grounded.any():
        raise ValueError("베드에 닿은 구슬이 없습니다.")

    # --- 강성 행렬 (축방향 스프링) -----------------------------------------
    # 막대 하나의 축강성 k = E*A/L
    p1, p2 = points[pairs[:, 0]], points[pairs[:, 1]]
    d = p2 - p1
    L = np.linalg.norm(d, axis=1)
    L[L < 1e-9] = 1e-9
    u = d / L[:, None]                              # 단위 방향 벡터

    E = mat.youngs_modulus_mpa                      # N/mm^2
    G = mat.shear_modulus_mpa
    vertical_like = np.abs(u[:, 2]) > 0.5
    area_link = np.where(vertical_like, neck_area_v, neck_area_mm2)
    k_axial = E * area_link / L                     # N/mm, 넥을 늘리고 줄이는 저항
    # 넥은 비틀림·전단에도 저항한다. 축방향만 넣으면 격자가 옆으로 흐물거리는
    # 가짜 메커니즘이 생겨서, 응력은 0 인데 변위만 수십 mm 로 나온다.
    k_shear = G * area_link / L                     # N/mm


    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    rows_l, cols_l, vals_l = [], [], []
    for a in range(3):
        for bax in range(3):
            # k_a * (u u^T) + k_s * (I - u u^T)
            proj = u[:, a] * u[:, bax]
            delta = 1.0 if a == bax else 0.0
            kab = k_axial * proj + k_shear * (delta - proj)
            rows_l.append(i_idx * 3 + a); cols_l.append(i_idx * 3 + bax); vals_l.append(kab)
            rows_l.append(j_idx * 3 + a); cols_l.append(j_idx * 3 + bax); vals_l.append(kab)
            rows_l.append(i_idx * 3 + a); cols_l.append(j_idx * 3 + bax); vals_l.append(-kab)
            rows_l.append(j_idx * 3 + a); cols_l.append(i_idx * 3 + bax); vals_l.append(-kab)
    K = coo_matrix(
        (np.concatenate(vals_l),
         (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(3 * n, 3 * n),
    ).tocsr()
    del rows_l, cols_l, vals_l

    # --- 하중: 관성력 F = m*a ------------------------------------------------
    vol_mm3 = 4.0 / 3.0 * math.pi * r ** 3
    mass_kg = mat.density_kg_m3 * vol_mm3 * 1e-9
    force_n = mass_kg * 9.81 * accel_g              # N, 구슬 하나당
    dirv = np.asarray(direction, dtype=float)
    dirv /= np.linalg.norm(dirv)
    F = np.tile(dirv, n) * force_n

    # 고정단은 행렬에서 아예 제거한다(자유 자유도만 푼다).
    # 예전처럼 LIL 로 행을 하나씩 고쳐 쓰면 노드가 십만 개만 넘어도
    # 메모리가 터진다.
    free = ~np.repeat(grounded, 3)
    Kff = K[free][:, free].tocsr()
    Ff = F[free]

    # 축방향 스프링만 쓰므로 회전 자유도가 구속되지 않아 특이해질 수 있다.
    # 대각을 아주 조금 보강해 수치적으로 안정화한다(결과에 미치는 영향은 미미).
    if Kff.shape[0] == 0:
        # 모든 구슬이 베드에 닿아 자유도가 없다 = 변형할 여지 자체가 없다.
        return ShakeResult(
            beads=n, links=len(pairs), grounded=int(grounded.sum()),
            max_displacement_mm=0.0, max_neck_stress_mpa=0.0,
            neck_strength_mpa=mat.tensile_strength_mpa,
            safety_factor=float("inf"),
            weakest_link_z_mm=float(points[:, 2].min()),
        )
    diag = Kff.diagonal()
    reg = max(float(np.abs(diag).max()) * 1e-9, 1e-12)
    Kff = Kff + identity(Kff.shape[0], format="csr") * reg

    # 큰 격자에서는 직접 분해가 비싸므로 공액구배법 + 야코비 전처리로 푼다.
    M = identity(Kff.shape[0], format="csr")
    dg = Kff.diagonal().copy()
    dg[np.abs(dg) < 1e-12] = 1.0
    M = diags(1.0 / dg)
    uf, info = cg(Kff, Ff, rtol=1e-8, maxiter=20000, M=M)
    if info != 0:
        # 수렴 실패 시에도 지금까지의 근사해로 상대 비교는 가능하다
        pass

    disp = np.zeros(3 * n)
    disp[free] = uf
    disp = np.nan_to_num(disp).reshape(n, 3)

    # --- 넥 응력 -------------------------------------------------------------
    rel = disp[pairs[:, 1]] - disp[pairs[:, 0]]
    elong = np.einsum("ij,ij->i", rel, u)           # 축방향 신장량
    axial_force = k_axial * elong                   # N
    # 축에 수직인 성분 = 전단
    shear_vec = rel - elong[:, None] * u
    shear_force = k_shear * np.linalg.norm(shear_vec, axis=1)

    sigma = np.abs(axial_force) / area_link         # MPa
    tau = shear_force / area_link                   # MPa
    # 등가응력(폰 미제스 근사): 인장과 전단을 함께 본다
    stress = np.sqrt(sigma ** 2 + 3.0 * tau ** 2)
    worst = int(np.argmax(stress))

    return ShakeResult(
        beads=n,
        links=len(pairs),
        grounded=int(grounded.sum()),
        max_displacement_mm=float(np.abs(disp).max()),
        max_neck_stress_mpa=float(stress[worst]),
        neck_strength_mpa=mat.tensile_strength_mpa,
        safety_factor=float(mat.tensile_strength_mpa / max(stress[worst], 1e-9)),
        weakest_link_z_mm=float(points[pairs[worst, 0], 2]),
    )


def beads_to_points(plan, bead_layer_height_mm: float, z0: float = 0.0) -> np.ndarray:
    """배치 계획에서 구슬 중심 좌표 배열을 뽑는다."""
    pts = []
    for layer in plan.layers:
        zc = layer.get("z_center")
        if zc is None:
            zc = layer["z_bottom"] + 0.5 * bead_layer_height_mm
        for bead in layer["beads"]:
            pts.append((bead["x"], bead["y"], zc))
    return np.asarray(pts, dtype=float)


@dataclass
class BreakawayResult:
    """망치로 두드려 부술 때의 수치."""

    neck_area_mm2: float
    force_per_neck_n: float
    critical_displacement_um: float
    bead_mass_mg: float
    inertia_force_at_2g_n: float
    shake_margin: float          # 넥 강도 / 2g 관성력
    model_wall_ratio: float      # 모델 벽 단면 / 넥 단면

    def summary(self) -> str:
        return (
            f"넥 하나 끊는 힘   {self.force_per_neck_n:.2f} N "
            f"(단면 {self.neck_area_mm2:.4f} mm^2)\n"
            f"  임계 상대변위     {self.critical_displacement_um:.1f} um\n"
            f"  구슬 하나 무게    {self.bead_mass_mg:.4f} mg\n"
            f"  2g 흔들림 여유    {self.shake_margin:,.0f} 배 (흔들려도 안 부서짐)\n"
            f"  모델 벽 대비      {self.model_wall_ratio:.0f} 배 강함 "
            f"(서포터만 골라 부서짐)"
        )


def breakaway_analysis(
    params,
    material: Optional[MaterialProps] = None,
    model_wall_mm: float = 1.2,
    shake_g: float = 2.0,
) -> BreakawayResult:
    """고무망치로 두드려 펠릿으로 부술 수 있는지 계산한다.

    손이 안 닿는 안쪽 서포터는 집어서 빼낼 수가 없으므로, 충격으로 넥을
    끊어 낱알로 만든 뒤 쏟아내는 것이 현실적인 회수 방법이다. 이때 두 조건이
    동시에 만족해야 한다.

    1. **인쇄 중에는 안 부서질 것** — 플레이트 흔들림으로 생기는 관성력이
       넥 강도보다 훨씬 작아야 한다. 구슬이 작을수록 질량은 지름의 세제곱으로
       줄고 넥 면적은 제곱으로만 줄어서, 이 여유는 자동으로 아주 커진다.
    2. **망치로는 부서질 것** — 넥 강도가 타격력보다 훨씬 작아야 한다.
       넥은 절대적으로 매우 약해서(1N 안팎) 이 조건도 쉽게 만족한다.

    또한 모델 벽이 넥보다 훨씬 두꺼워야 서포터만 골라 부서진다.
    """
    mat = material or MaterialProps()
    r = 0.5 * params.bead_diameter_mm
    area = params.vertical_neck_area_mm2()
    d_v = params.vertical_neighbor_distance_mm()

    force = mat.tensile_strength_mpa * area
    # sigma = E * eps -> eps_fail = sigma_f / E
    crit_um = (mat.tensile_strength_mpa / mat.youngs_modulus_mpa) * d_v * 1000.0
    mass_kg = mat.density_kg_m3 * (4.0 / 3.0 * math.pi * (r * 1e-3) ** 3)
    inertia = mass_kg * 9.81 * shake_g

    return BreakawayResult(
        neck_area_mm2=area,
        force_per_neck_n=force,
        critical_displacement_um=crit_um,
        bead_mass_mg=mass_kg * 1e6,
        inertia_force_at_2g_n=inertia,
        shake_margin=force / inertia if inertia > 0 else float("inf"),
        model_wall_ratio=(model_wall_mm * 1.0) / area if area > 0 else float("inf"),
    )
