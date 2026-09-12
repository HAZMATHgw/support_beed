# -*- coding: utf-8 -*-
"""입력 검증.

이 도구가 조용히 틀린 결과를 내는 경우가 많아서 따로 모아 둔다. 예를 들어
``xy_clearance`` 를 음수로 주면 서포터가 모델을 파고든 채로 구슬이 오히려 더
많이 생성되는데, 예전에는 아무 경고 없이 그냥 통과했다.

원칙:

- 물리적으로 말이 안 되는 값(지름 0, 겹침 100%, 음수 여유)은 **에러**로 막는다.
- 계산은 되지만 의도한 결과가 아닐 값(겹침 과다, 여백 과다)은 **경고**만 낸다.
- 에러 메시지는 무엇을 어떻게 고쳐야 하는지까지 적는다.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from .params import SupportBeadParams, SupportGenParams


class InvalidParameterError(ValueError):
    """파라미터 값이 물리적으로 성립하지 않을 때."""


class InvalidModelError(ValueError):
    """입력 메쉬 자체가 처리 불가능할 때."""


class TooManyBeadsError(RuntimeError):
    """구슬 수가 감당 못 할 정도로 많을 때. 멈추기 전에 미리 막는다."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidParameterError(message)


def validate_bead_params(params: SupportBeadParams, label: str = "구슬") -> None:
    """구슬 기하 파라미터 검사."""
    _require(
        params.bead_diameter_mm > 0,
        f"{label} 지름이 {params.bead_diameter_mm}mm 입니다. 0보다 커야 합니다.",
    )
    _require(
        params.lattice_overlap_ratio < 1.0,
        f"{label} 겹침(delta)이 {params.lattice_overlap_ratio} 입니다. "
        f"1 이상이면 구슬 간격(pitch)이 0 이하가 되어 격자를 만들 수 없습니다. "
        f"0.04~0.08 을 권장합니다.",
    )
    _require(
        params.lattice_overlap_ratio >= 0.0,
        f"{label} 겹침(delta)이 {params.lattice_overlap_ratio} 입니다. "
        f"음수면 구슬끼리 닿지 않아 서포터가 무너집니다. 0 이상이어야 합니다.",
    )
    _require(
        params.edge_margin_ratio >= 0,
        f"{label} 가장자리 여백 비율이 음수입니다({params.edge_margin_ratio}). "
        f"구슬이 영역 밖으로 나가 모델과 충돌합니다.",
    )
    _require(
        params.segment_ratio >= 0,
        f"{label} 압출 선분 비율이 음수입니다({params.segment_ratio}).",
    )
    _require(
        params.flow_multiplier > 0,
        f"{label} 압출량 배수가 {params.flow_multiplier} 입니다. 0보다 커야 합니다.",
    )

    for label_dir, value in (("가로", params.lateral_overlap()),
                             ("세로", params.vertical_overlap())):
        _require(
            0.0 <= value < 1.0,
            f"{label} {label_dir} 겹침이 {value} 입니다. 0 이상 1 미만이어야 합니다.",
        )
    # 가로를 너무 눌러 붙이면 세로로 층을 쌓을 수 없다(h 가 허수가 된다).
    pitch = params.pitch_mm()
    d_v = params.vertical_neighbor_distance_mm()
    if params.stagger_layers and d_v * d_v - pitch * pitch / 3.0 <= 0:
        raise InvalidParameterError(
            f"{label} 가로 겹침({params.lateral_overlap():.3f})이 세로 "
            f"겹침({params.vertical_overlap():.3f})에 비해 너무 큽니다. "
            f"층을 쌓을 높이가 나오지 않습니다. 가로를 줄이거나 세로를 늘리세요."
        )

    if params.lattice_overlap_ratio > 0.12:
        warnings.warn(
            f"{label} 겹침(delta)이 {params.lattice_overlap_ratio:.2f} 로 큽니다. "
            f"구슬이 뭉쳐 통짜에 가까워지고 펠릿으로 부서지지 않습니다.",
            stacklevel=2,
        )
    if params.stagger_period < 2:
        warnings.warn(
            f"{label} 쌓기 순환이 {params.stagger_period} 입니다. "
            f"2(HCP) 로 올려서 계산합니다.",
            stacklevel=2,
        )


def validate_gen_params(gen: SupportGenParams) -> None:
    """서포터 영역 탐지 파라미터 검사."""
    _require(
        math.isfinite(gen.branch_angle_deg) and 0 <= gen.branch_angle_deg < 90,
        "트리 가지 각도는 수직 기준 0 이상 90도 미만이어야 합니다.",
    )
    for name, value in (
        ("트리 접점 간격", gen.tree_contact_spacing_mm),
        ("트리 병합 거리", gen.branch_merge_distance_mm),
    ):
        _require(value is None or (math.isfinite(value) and value > 0),
                 f"{name}은 유한한 양수(mm)여야 합니다. 비워두면 자동입니다.")
    _require(math.isfinite(gen.contact_z_gap_mm) and gen.contact_z_gap_mm >= 0,
             "모델과의 Z 간격(mm)은 유한한 0 이상의 값이어야 합니다.")
    _require(gen.max_beads is None or (isinstance(gen.max_beads, int) and gen.max_beads > 0),
             "구슬 수 상한은 양의 정수여야 합니다.")
    _require(
        gen.layer_height_mm > 0,
        f"구슬 층높이가 {gen.layer_height_mm}mm 입니다. "
        f"보통 구슬 파라미터에서 자동 계산되므로, 직접 지정했다면 지우세요.",
    )
    _require(
        0 < gen.overhang_angle_deg < 90,
        f"오버행 각도가 {gen.overhang_angle_deg}도 입니다. "
        f"0보다 크고 90보다 작아야 합니다(권장 40~60).",
    )
    _require(
        gen.xy_clearance_mm >= 0,
        f"모델과의 XY 여유가 {gen.xy_clearance_mm}mm 입니다. "
        f"음수면 서포터가 모델을 파고들어 떼어낼 수 없게 됩니다.",
    )
    _require(
        gen.contact_layers >= 1,
        f"인터페이스 층 수가 {gen.contact_layers} 입니다. 1 이상이어야 합니다.",
    )
    _require(
        gen.contact_z_gap_layers >= 0,
        f"모델과의 Z 간격이 {gen.contact_z_gap_layers}층 입니다. "
        f"음수면 서포터가 모델 안으로 파고듭니다.",
    )
    _require(
        math.isfinite(gen.tree_trunk_slenderness) and gen.tree_trunk_slenderness > 0,
        f"트렁크 세장비 상한이 {gen.tree_trunk_slenderness} 입니다. 0보다 커야 합니다"
        f"(권장 5~12, 작을수록 굵고 튼튼).",
    )
    _require(
        gen.tree_brace_distance_mm is None or
        (math.isfinite(gen.tree_brace_distance_mm) and gen.tree_brace_distance_mm > 0),
        f"트렁크 연결 최대 거리가 {gen.tree_brace_distance_mm}mm 입니다. "
        f"0보다 커야 합니다(비우면 자동).",
    )
    _require(
        gen.brim_mm >= 0,
        f"바닥 브림 폭이 음수입니다({gen.brim_mm}).",
    )
    _require(
        gen.brim_height_mm >= 0,
        f"바닥 브림 높이가 음수입니다({gen.brim_height_mm}).",
    )
    _require(
        gen.flare_mm_per_layer >= 0,
        f"서포터 확장량이 음수입니다({gen.flare_mm_per_layer}). "
        f"아래로 갈수록 좁아지면 흔들림에 넘어집니다.",
    )
    _require(
        gen.min_island_area_mm2 >= 0,
        f"최소 조각 면적이 음수입니다({gen.min_island_area_mm2}).",
    )
    _require(
        gen.solid_first_layers >= 0,
        f"첫 통판 층 수가 음수입니다({gen.solid_first_layers}).",
    )
    if gen.detection_layer_height_mm is not None:
        _require(
            gen.detection_layer_height_mm > 0,
            f"오버행 탐지 두께가 {gen.detection_layer_height_mm}mm 입니다. "
            f"0보다 커야 합니다. 비워두면 자동으로 정해집니다.",
        )


def validate_mesh(mesh) -> None:
    """입력 메쉬가 슬라이싱 가능한 상태인지 검사."""
    faces = getattr(mesh, "faces", None)
    if mesh is None or faces is None or len(faces) == 0:
        raise InvalidModelError(
            "모델에 삼각형이 없습니다. 정점만 있고 면이 없거나, 파일이 비어 "
            "있거나 형식이 잘못됐는지 확인해 주세요."
        )
    raw_verts = getattr(mesh, "vertices", None)
    if raw_verts is None or len(raw_verts) == 0:
        raise InvalidModelError("모델에 정점이 없습니다.")
    verts = np.asarray(raw_verts, dtype=float)
    if not np.isfinite(verts).all():
        n_bad = int((~np.isfinite(verts)).any(axis=1).sum())
        raise InvalidModelError(
            f"모델 정점 {n_bad}개의 좌표가 NaN 또는 무한대입니다. "
            f"3D 편집 도구에서 메쉬를 정리(repair)한 뒤 다시 시도하세요."
        )
    raw_extents = getattr(mesh, "extents", None)
    if raw_extents is None:
        raise InvalidModelError(
            "모델의 크기를 계산할 수 없습니다. 메쉬가 손상됐는지 확인해 주세요."
        )
    extents = np.asarray(raw_extents, dtype=float)
    if not np.isfinite(extents).all() or extents[2] <= 0:
        raise InvalidModelError(
            f"모델 높이가 {extents[2] if np.isfinite(extents).all() else '비정상'} 입니다. "
            f"두께가 있는 입체여야 합니다."
        )


def check_model_scale(mesh, contact: SupportBeadParams) -> None:
    """모델이 구슬에 비해 너무 작으면 미리 알려준다."""
    smallest = float(np.min(mesh.extents))
    if smallest < contact.bead_diameter_mm:
        warnings.warn(
            f"모델의 가장 짧은 변이 {smallest:.2f}mm 인데 구슬 지름은 "
            f"{contact.bead_diameter_mm:.2f}mm 입니다. 구슬이 들어갈 자리가 없어 "
            f"서포터가 거의 생성되지 않습니다. --bead-diameter 를 줄이세요.",
            stacklevel=2,
        )


def fillable_fraction(support_regions, bead_diameter_mm: float) -> float:
    """서포터 영역 중 구슬이 실제로 들어갈 수 있는 면적 비율.

    영역을 구슬 반지름만큼 깎아도 남는 부분에만 구슬 중심을 놓을 수 있다.
    구슬이 영역 폭보다 크면 아무리 격자를 잘 깔아도 채울 수가 없다.
    """
    total = fits = 0.0
    for region in support_regions:
        if region is None or region.is_empty:
            continue
        total += region.area
        eroded = region.buffer(-0.5 * bead_diameter_mm)
        if not eroded.is_empty:
            fits += eroded.area
    return (fits / total) if total > 0 else 1.0


def check_bead_fits_regions(support_regions, contact: SupportBeadParams,
                            nozzle_diameter_mm: float,
                            threshold: float = 0.8) -> Optional[str]:
    """구슬이 서포터 영역에 비해 너무 크면 경고 문구를 만든다."""
    frac = fillable_fraction(support_regions, contact.bead_diameter_mm)
    if frac >= threshold:
        return None
    return (
        f"구슬 지름({contact.bead_diameter_mm:.2f}mm)이 서포터 영역에 비해 큽니다. "
        f"영역의 {(1 - frac) * 100:.0f}%가 구슬보다 좁아서 채울 수 없고, "
        f"그만큼 구슬이 끊기거나 빈 곳이 생깁니다.\n"
        f"  해결: --bead-diameter 를 줄이세요"
        f"(이 노즐의 인쇄 가능 최소는 약 "
        f"{nozzle_diameter_mm * 0.35:.2f}mm 입니다). 모델이 노즐에 비해 너무 "
        f"작으면 더 가는 노즐이 필요합니다."
    )


def quick_overhang_estimate(mesh, gen: SupportGenParams,
                            contact: SupportBeadParams) -> int:
    """슬라이싱 없이 삼각형 법선만으로 몇 초 안에 구슬 수를 거칠게 가늠한다.

    정밀한 추정(:func:`estimate_bead_count`)은 전체 슬라이싱과 영역 계산을
    거쳐야 해서, 나뭇가지처럼 복잡한 모델(삼각형 190만 개)에서는 그 계산
    자체에 68초가 걸렸다. 그동안 브라우저·프록시가 연결을 끊어서 "서버가
    죽었다"로 보였다. 여기서는 그 판단을 몇 초 안에 앞당겨서 낸다.

    아래로 향한 삼각형의 실제 면적(경사 포함)을 오버행 면적으로 보고,
    모델 높이의 절반 정도를 채운다고 가정하는 아주 거친 상한 추정이다.
    실측 오차는 모델에 따라 최대 2.5배 정도 나므로, 최종 판정을 대신하지
    않고 "확실히 과한 경우"만 미리 걸러내는 안전판으로만 쓴다.
    """
    if len(mesh.faces) == 0:
        return 0
    normals = mesh.face_normals
    areas = mesh.area_faces
    ang = math.radians(max(1.0, min(89.0, gen.overhang_angle_deg)))
    down = normals[:, 2] < -math.sin(ang)
    overhang_area = float(areas[down].sum())
    if overhang_area <= 0:
        return 0

    pitch = contact.pitch_mm()
    if pitch <= 0:
        return 0
    cell_area = math.sqrt(3.0) / 2.0 * pitch * pitch
    avg_height = float(mesh.extents[2]) * 0.5
    n_cols = overhang_area / cell_area
    n_layers = avg_height / max(gen.layer_height_mm, 1e-6)
    return int(n_cols * n_layers)


def guard_quick_estimate(quick: int, max_beads: int, safety_factor: float = 3.0) -> None:
    """빠른 추정이 상한을 한참 넘으면(안전계수 반영) 정밀 계산 전에 먼저 막는다.

    빠른 추정은 최대 2.5배까지 어긋날 수 있으므로, 상한의 ``safety_factor``
    배를 넘을 때만 차단한다. 애매한 경우는 그냥 통과시켜 정밀 계산이
    최종 판단하게 한다.
    """
    if quick > max_beads * safety_factor:
        raise TooManyBeadsError(
            f"구슬이 최소 약 {quick:,}개 필요할 것으로 보입니다(상한 {max_beads:,}개). "
            f"정밀 계산 없이 미리 알려드립니다 — 이대로면 계산에만 1분 넘게 걸리고 "
            f"결국 메모리가 부족해 멈춥니다.\n"
            f"해결 방법: --bead-diameter 를 키우거나, --overhang-angle 을 낮춰 "
            f"서포터 범위를 줄이거나, --max-beads 로 상한을 올리세요."
        )


def estimate_bead_count(support_regions, gen: SupportGenParams,
                        contact: SupportBeadParams) -> int:
    """생성될 구슬 수를 미리 어림한다(메모리 폭발 방지용)."""
    pitch = contact.pitch_mm()
    if pitch <= 0:
        return 0
    cell = math.sqrt(3.0) / 2.0 * pitch * pitch
    total_area = 0.0
    for region in support_regions:
        if region is not None and not region.is_empty:
            total_area += region.area
    return int(total_area / cell) if cell > 0 else 0


#: 구 표면 세분화 단계별 (정점, 면) 수
_SPHERE_SIZE = {0: (12, 20), 1: (42, 80), 2: (162, 320), 3: (642, 1280)}


#: 메쉬를 합칠 때 원본과 사본이 동시에 존재하므로 피크 사용량은 몇 배가 된다.
_PEAK_FACTOR = 3


def available_memory_bytes(default: int = 2_000_000_000) -> int:
    """지금 쓸 수 있는 메모리. 못 읽으면 보수적인 기본값."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return default


def bead_budget(detail: int, memory_budget_bytes: Optional[int] = None) -> int:
    """메모리 예산 안에서 만들 수 있는 구슬 수.

    구슬 하나가 차지하는 메모리는 세분화 단계에 따라 크게 달라진다
    (detail 1 은 detail 0 의 약 4배). 고정 상한을 쓰면 detail 0 에서는
    지나치게 빡빡하고 detail 2 에서는 상한을 지켜도 메모리가 터진다.
    """
    if memory_budget_bytes is None:
        # 가용 메모리를 다 쓰면 다른 프로세스까지 죽는다. 60%만 쓴다.
        memory_budget_bytes = int(available_memory_bytes() * 0.6)
    verts, faces = _SPHERE_SIZE.get(max(0, min(3, detail)), _SPHERE_SIZE[1])
    per_bead = (verts + faces) * 3 * 8 * _PEAK_FACTOR
    return max(1000, memory_budget_bytes // per_bead)


def guard_bead_count(estimated: int, max_beads: int) -> None:
    """구슬이 너무 많으면 멈추기 전에 막고, 무엇을 바꿔야 하는지 알려준다."""
    if estimated > max_beads:
        raise TooManyBeadsError(
            f"구슬이 약 {estimated:,}개 필요합니다(상한 {max_beads:,}개). "
            f"이대로 진행하면 메모리가 부족해 멈춥니다.\n"
            f"해결 방법: --bead-diameter 를 키우거나, --overhang-angle 을 낮춰 "
            f"서포터 범위를 줄이거나, --max-beads 로 상한을 올리세요."
        )
