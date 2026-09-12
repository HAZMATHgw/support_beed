# -*- coding: utf-8 -*-
"""모델과 노즐에 맞는 구슬 크기 자동 선택.

구슬 크기는 두 가지가 맞선다.

- **크면**: 구슬 수가 적어 인쇄가 빠르지만, 서포터 영역이 구슬보다 좁은 곳이
  많아져서 구슬이 안 들어가고 사슬이 끊긴다.
- **작으면**: 구석구석 채워져 튼튼하지만, 같은 부피를 채우는 데 구슬 수가
  지름의 세제곱에 반비례해 늘어난다(지름 절반 -> 개수 8배).

여기서는 "서포터 영역 중 구슬이 실제로 들어갈 수 있는 면적 비율"을 품질
지표로 삼는다. 이 값은 최종 연결 품질을 잘 예측하면서도, 구슬을 실제로
배치해 보지 않고 영역만으로 계산할 수 있어 빠르다.

배 모델(31x60x48mm) 기준 실측:

    구슬     채움가능    구슬수    덩어리   이웃6개이상
    2.50mm    49.1%       507      4      71.6%
    1.75mm    62.2%     3,016      2      85.0%
    1.25mm    72.0%    10,708      2      93.9%
    1.00mm    77.1%    20,779      2      96.1%

전략은 "목표 품질을 만족하는 것 중 가장 큰 구슬"이다. 필요 이상으로 작게
잡아 인쇄 시간을 낭비하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from .params import SupportBeadParams, SupportGenParams
from .regions import build_support_regions, detect_overhangs
from .slicing import clean, slice_model
from .validation import (fillable_fraction, validate_bead_params,
                         validate_gen_params, validate_mesh)


@dataclass
class TuningCandidate:
    bead_diameter_mm: float
    fillable_fraction: float
    estimated_beads: int
    layer_height_mm: float


@dataclass
class TuningResult:
    chosen: TuningCandidate
    candidates: List[TuningCandidate]
    target: float
    met_target: bool
    #: 남은 틈을 메우는 데 쓸 수 있는 더 작은 구슬 지름들
    filler_diameters: List[float] = None
    tree_enabled: bool = False

    def summary(self) -> str:
        lines = [
            f"자동 선택: 기본 구슬 {self.chosen.bead_diameter_mm:.2f}mm "
            f"(층높이 {self.chosen.layer_height_mm:.3f}mm)",
            f"  기본 구슬이 채우는 영역 {self.chosen.fillable_fraction * 100:.0f}% "
            f"/ 예상 {self.chosen.estimated_beads:,}개",
        ]
        if self.tree_enabled:
            lines.append("  트리 예상 개수는 가지 병합·몸통 보강 전 근사치입니다.")
        elif self.filler_diameters:
            sizes = ", ".join(f"{d:.2f}mm" for d in self.filler_diameters)
            lines.append(f"  남은 틈은 더 작은 구슬로 메움: {sizes}")
        else:
            lines.append(
                "  ! 더 작은 구슬을 쓸 여유가 없습니다"
                "(기본 구슬이 이미 인쇄 가능 최소 크기). 틈이 남습니다."
            )
        if not self.met_target:
            lines.append(
                f"  ! 목표 품질({self.target * 100:.0f}%)에 도달하지 못했습니다. "
                f"모델이 노즐에 비해 작습니다. 더 가는 노즐이 필요합니다."
            )
        return "\n".join(lines)


def auto_tune_bead_diameter(
    mesh,
    gen: SupportGenParams,
    base_contact: SupportBeadParams,
    target_fill: float = 0.75,
    max_beads: Optional[int] = None,
    steps: int = 8,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TuningResult:
    """모델과 노즐에 맞는 구슬 지름을 고른다.

    ``target_fill`` 이상을 만족하는 후보 중 **가장 큰** 구슬을 고른다.
    ``max_beads`` 를 주면 그 개수를 넘는 후보는 인쇄 시간이 과해 제외한다.
    아무 후보도 목표를 만족하지 못하면 가장 품질이 좋은 후보를 고르고
    ``met_target=False`` 로 알린다.
    """
    validate_mesh(mesh)
    validate_gen_params(gen)
    validate_bead_params(base_contact)
    report = progress_callback or (lambda stage: None)
    nozzle = gen.nozzle_diameter_mm
    min_bead = gen.min_bead_diameter_mm or (nozzle * gen.min_bead_to_nozzle_ratio)
    max_bead = max(min_bead, nozzle * 0.5)
    if steps < 2:
        steps = 2

    # 큰 것부터 작은 것까지 균등하게 후보를 만든다.
    diameters = list(dict.fromkeys(
        max_bead - (max_bead - min_bead) * i / (steps - 1) for i in range(steps)
    ))

    # 형상 탐지는 후보 지름과 분리한다. 예전에는 후보마다 전체 메시를
    # 다시 자르고 아래층까지 영역을 확장해, 작은 노즐에서 수 분씩 걸렸다.
    # 모든 후보를 같은 단면으로 비교하고 명시한 탐지 해상도도 지킨다.
    det_h = gen.detection_layer_height_mm or min(0.4, gen.layer_height_mm)
    det_h = max(det_h, float(mesh.extents[2]) / gen.max_detection_layers)
    report("구슬 크기 자동 선택 · 모델 단면 계산 (1회)")
    det_slices, _ = slice_model(mesh, det_h, gen.max_detection_layers)
    report("구슬 크기 자동 선택 · 지지 영역 분석")
    if gen.tree_enabled:
        # 트리는 오버행 접점을 가지로 잇는다. 격자처럼 전체 지지 부피를
        # 만들 필요가 없으며, 그 부피로 개수를 추정하면 크게 부풀려진다.
        analysis_gen = replace(gen, removal_opening_mm=max_bead)
        support = detect_overhangs(det_slices, analysis_gen, det_h)
    else:
        support, _ = build_support_regions(det_slices, gen, det_h)
    regions = [clean(s) for s in support]
    total_area = sum(r.area for r in regions if not r.is_empty)
    # 트리 개수는 병합/몸통 보강 전 수직 가지 길이의 근사치다.
    # 실제 배치 개수나 최종 연결 품질의 보증으로 쓰지는 않는다.
    height_weighted_area = sum(
        r.area * (i + 0.5) * det_h for i, r in enumerate(regions)
        if not r.is_empty
    )

    candidates: List[TuningCandidate] = []
    for index, d in enumerate(diameters, 1):
        report(f"구슬 크기 자동 선택 · 후보 {index}/{len(diameters)} ({d:.3f} mm)")
        probe = replace(base_contact, bead_diameter_mm=d)
        layer_h = probe.layer_height_mm()
        if layer_h <= 0:
            continue
        frac = fillable_fraction(regions, d)

        # 구슬 수 어림: 채울 수 있는 면적 / 격자 셀 면적
        pitch = probe.pitch_mm()
        cell = math.sqrt(3.0) / 2.0 * pitch * pitch
        area = total_area * frac
        # 탐지 층 기준 면적이므로 구슬 층 간격으로 환산한다
        scale = det_h / layer_h if layer_h > 0 else 1.0
        if gen.tree_enabled:
            spacing = gen.tree_contact_spacing_mm or 4.0 * d
            est = int(height_weighted_area * frac / (spacing * spacing * layer_h))
        else:
            est = int(area * scale / cell) if cell > 0 else 0

        candidates.append(TuningCandidate(d, frac, est, layer_h))

    if not candidates:
        raise RuntimeError("구슬 크기 후보를 만들지 못했습니다.")

    # 기본 구슬이 인쇄 가능 최소 크기에 딱 붙으면 세분 구슬을 쓸 여유가
    # 없어서 남은 틈을 못 메운다. 최소 한 세대는 내려갈 수 있도록,
    # 여유가 있는 후보를 우선한다.
    headroom = min_bead / max(gen.fill_shrink, 1e-6)
    with_headroom = [c for c in candidates
                     if c.bead_diameter_mm >= headroom - 1e-9]

    # 세분 충전을 감안한 유효 채움률.
    #
    # 기본 구슬만으로 목표를 못 채워도, 남은 틈을 더 작은 구슬로 어느 정도
    # 메울 수 있다. 다만 실측해 보면 그 효과가 크지 않다. 배 모델에서
    # 기본 2.5mm + 세분(구슬 1,598개)은 오버행면까지 4.97mm 였고,
    # 기본 1.43mm + 세분(7,481개)은 3.78mm 로 더 좋았다.
    # 세분 구슬은 격자가 어긋나 이웃과 잘 물리지 않기 때문이다.
    # 따라서 보너스는 아주 보수적으로만 준다.
    def effective_fill(cand: TuningCandidate) -> float:
        frac = cand.fillable_fraction
        if gen.tree_enabled:
            # 트리 생성은 격자의 세분 충전을 쓰지 않는다. 없는 보너스로
            # 목표 달성을 보고하거나 작은 접점용 구슬을 배제하지 않는다.
            return frac
        d = cand.bead_diameter_mm
        bonus = 0.0
        for _ in range(gen.fill_generations):
            d *= gen.fill_shrink
            if d < min_bead:
                break
            bonus += (1.0 - frac) * 0.05
        return min(frac + bonus, 1.0)

    usable = [c for c in candidates
              if effective_fill(c) >= target_fill
              and (max_beads is None or c.estimated_beads <= max_beads)]
    usable_hr = [c for c in usable if c in with_headroom] if not gen.tree_enabled else []
    if usable_hr:
        usable = usable_hr
    if usable:
        chosen = max(usable, key=lambda c: c.bead_diameter_mm)
        met = True
    else:
        pool = [c for c in candidates
                if max_beads is None or c.estimated_beads <= max_beads]
        if pool:
            # 목표 품질만 못 맞춘 경우: 그 안에서 품질이 가장 좋은 것
            chosen = max(pool, key=lambda c: c.fillable_fraction)
        else:
            # 개수 상한조차 만족하는 후보가 없다. 품질을 좇아 더 작은 구슬을
            # 고르면 개수가 더 늘어나 상한에서 더 멀어진다. 상한에 가장 가까운
            # (= 구슬이 가장 적은) 것을 고르는 편이 사용자 의도에 맞다.
            chosen = min(candidates, key=lambda c: c.estimated_beads)
        met = False

    # 남은 틈을 메울 더 작은 구슬 지름을 미리 계산해 둔다.
    # 기본 구슬이 이미 최소 크기면 여유가 없어 틈이 그대로 남는다.
    fillers: List[float] = []
    d = chosen.bead_diameter_mm
    for _ in range(0 if gen.tree_enabled else gen.fill_generations):
        d *= gen.fill_shrink
        if d < min_bead:
            break
        fillers.append(round(d, 3))

    return TuningResult(chosen, candidates, target_fill, met, fillers, gen.tree_enabled)
