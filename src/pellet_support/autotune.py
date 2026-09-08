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
from typing import List, Optional

from .params import SupportBeadParams, SupportGenParams
from .regions import build_support_regions
from .slicing import clean, slice_model
from .validation import fillable_fraction


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

    def summary(self) -> str:
        lines = [
            f"자동 선택: 구슬 {self.chosen.bead_diameter_mm:.2f}mm "
            f"(층높이 {self.chosen.layer_height_mm:.3f}mm)",
            f"  채울 수 있는 영역 {self.chosen.fillable_fraction * 100:.0f}% "
            f"/ 예상 구슬 {self.chosen.estimated_beads:,}개",
        ]
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
) -> TuningResult:
    """모델과 노즐에 맞는 구슬 지름을 고른다.

    ``target_fill`` 이상을 만족하는 후보 중 **가장 큰** 구슬을 고른다.
    ``max_beads`` 를 주면 그 개수를 넘는 후보는 인쇄 시간이 과해 제외한다.
    아무 후보도 목표를 만족하지 못하면 가장 품질이 좋은 후보를 고르고
    ``met_target=False`` 로 알린다.
    """
    nozzle = gen.nozzle_diameter_mm
    min_bead = gen.min_bead_diameter_mm or (nozzle * gen.min_bead_to_nozzle_ratio)
    max_bead = max(min_bead, nozzle * 0.5)
    if steps < 2:
        steps = 2

    # 큰 것부터 작은 것까지 균등하게 후보를 만든다.
    diameters = [
        max_bead - (max_bead - min_bead) * i / (steps - 1) for i in range(steps)
    ]

    candidates: List[TuningCandidate] = []
    for d in diameters:
        probe = replace(base_contact, bead_diameter_mm=d)
        layer_h = probe.layer_height_mm()
        if layer_h <= 0:
            continue
        probe_gen = replace(gen, layer_height_mm=layer_h)
        det_h = probe_gen.detection_height_mm() if hasattr(
            probe_gen, "detection_height_mm") else min(0.4, layer_h)
        try:
            det_slices, _ = slice_model(mesh, det_h, gen.max_detection_layers)
            support, _ = build_support_regions(det_slices, probe_gen, det_h)
        except Exception:
            continue
        regions = [clean(s) for s in support]
        frac = fillable_fraction(regions, d)

        # 구슬 수 어림: 채울 수 있는 면적 / 격자 셀 면적
        pitch = probe.pitch_mm()
        cell = math.sqrt(3.0) / 2.0 * pitch * pitch
        area = sum(r.area for r in regions if not r.is_empty) * frac
        # 탐지 층 기준 면적이므로 구슬 층 간격으로 환산한다
        scale = det_h / layer_h if layer_h > 0 else 1.0
        est = int(area * scale / cell) if cell > 0 else 0

        candidates.append(TuningCandidate(d, frac, est, layer_h))

    if not candidates:
        raise RuntimeError("구슬 크기 후보를 만들지 못했습니다.")

    usable = [c for c in candidates
              if c.fillable_fraction >= target_fill
              and (max_beads is None or c.estimated_beads <= max_beads)]
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

    return TuningResult(chosen, candidates, target_fill, met)
