# -*- coding: utf-8 -*-
"""슬라이싱부터 메쉬까지 한 번에 묶는 진입점."""

from __future__ import annotations

import math

import warnings
from dataclasses import replace
from typing import Dict, NamedTuple, Optional, Tuple

import trimesh
from shapely.ops import unary_union

from .meshing import BeadPlan, deduplicate_layer_beads, plan_beads, plan_to_mesh
from .params import (
    SupportBeadParams,
    SupportBeadRegion,
    SupportGenParams,
    support_bead_body_params,
    support_bead_contact_params,
    unify_lattice,
)
from .regions import build_support_regions
from .repair import (
    analyze_connectivity,
    prune_weakly_connected,
    prune_orphan_clusters,
    repair_connectivity,
)
from .stability import prune_unsupported_beads
from .slicing import clean, slice_model
from .validation import (
    bead_budget,
    guard_quick_estimate,
    quick_overhang_estimate,
    check_bead_fits_regions,
    InvalidParameterError,
    check_model_scale,
    estimate_bead_count,
    guard_bead_count,
    validate_bead_params,
    validate_gen_params,
    validate_mesh,
)


class SupportResult(NamedTuple):
    mesh: trimesh.Trimesh
    plan: Optional[BeadPlan]
    slices: list


def make_params(
    nozzle_diameter_mm: float = 1.0,
    bead_diameter_mm: Optional[float] = None,
    overlap: Optional[float] = None,
    lateral_overlap: Optional[float] = None,
    vertical_overlap: Optional[float] = None,
    body_bead_ratio: float = 0.97,
    stagger_period: int = 3,
    straight_columns: bool = False,
    segment_ratio: Optional[float] = None,
    edge_margin_ratio: Optional[float] = None,
) -> Tuple[SupportBeadParams, SupportBeadParams]:
    """CLI 와 라이브러리가 공유하는 파라미터 조립 로직.

    ``bead_diameter_mm`` 을 비워두면 노즐 지름의 절반을 기본값으로 쓴다
    (표준 FDM 이 층높이를 노즐 지름의 25~75% 로 쓰는 것과 같은 논리).
    노즐이 굵어도 구를 그만큼 굵게 만들 필요는 없다 — 오히려 굵게 만들면
    gap/interface/edge_margin 이 전부 비드 지름에 비례해 커져서, 굵은
    펠릿 노즐일수록 서포터 위쪽에 큰 사각지대가 생긴다.
    """
    if nozzle_diameter_mm <= 0:
        raise InvalidParameterError(
            f"노즐 지름이 {nozzle_diameter_mm}mm 입니다. 0보다 커야 합니다."
        )
    if bead_diameter_mm is not None and bead_diameter_mm <= 0:
        raise InvalidParameterError(
            f"구슬 지름이 {bead_diameter_mm}mm 입니다. 0보다 커야 합니다. "
            f"자동값(노즐의 절반)을 쓰려면 비워 두세요."
        )
    if not (0 < body_bead_ratio <= 1):
        raise InvalidParameterError(
            f"몸통 구슬 비율이 {body_bead_ratio} 입니다. 0보다 크고 1 이하여야 합니다. "
            f"1을 넘으면 몸통 구슬이 인터페이스보다 커져서 공유 격자가 깨집니다."
        )

    contact = support_bead_contact_params(nozzle_diameter_mm, bead_diameter_mm)
    body = support_bead_body_params(nozzle_diameter_mm, body_bead_ratio, bead_diameter_mm)
    if overlap is not None:
        contact = replace(contact, lattice_overlap_ratio=overlap)
    # 비등방 충전: 가로만 강하게, 세로는 약하게.
    # 좌우 흔들림은 '가로 넥'이 버티고, 세로 넥은 약할수록 펠릿으로 잘 부서진다.
    if lateral_overlap is not None or vertical_overlap is not None:
        aniso = {}
        if lateral_overlap is not None:
            aniso["lateral_overlap_ratio"] = lateral_overlap
        if vertical_overlap is not None:
            aniso["vertical_overlap_ratio"] = vertical_overlap
        contact = replace(contact, **aniso)
        body = replace(body, **aniso)

    over = {"stagger_period": max(2, stagger_period)}
    if segment_ratio is not None:
        over["segment_ratio"] = segment_ratio
    if edge_margin_ratio is not None:
        over["edge_margin_ratio"] = edge_margin_ratio
    if straight_columns:
        over["stagger_layers"] = False

    contact = replace(contact, **over)
    body = replace(body, **over)
    contact, body = unify_lattice(contact, body)

    # 여기서 바로 검증한다. 나중에 검증하면 pitch 가 음수인 상태로 화면에
    # 출력되고 나서야 에러가 나서, 사용자가 이상한 숫자를 먼저 보게 된다.
    validate_bead_params(contact, "인터페이스 구슬")
    validate_bead_params(body, "몸통 구슬")
    return contact, body


def _generate_tree_support(
    mesh,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    det_slices,
    det_h: float,
    z0: float,
    detail: int,
    verbose: bool,
) -> "SupportResult":
    """나뭇가지 골격을 만들고 그것을 구슬로 표현한다.

    접촉점 -> 기울기 한계 안에서 V 자 병합하며 성장(베드 도달 가능 영역 회피)
    -> 트렁크 굵기(짐 + 세장비) -> 최밀충전 구슬 다발 -> 이웃 트렁크 연결
    -> 3D 관통 정리 -> 구조 점검(연결 덩어리, 받침 없는 구슬).
    """
    import math as _math

    from shapely.geometry import Polygon as _Polygon

    from .skeleton import (
        add_bracing,
        assign_hierarchical_radii,
        extract_contact_points,
        grow_branches,
        prune_floating,
        settle_collisions,
        skeleton_to_bead_seeds,
        structure_report,
    )

    heights = [z0 + (i + 0.5) * det_h for i in range(len(det_slices))]

    # 오버행: 아래층을 부풀린 것에서 삐져나온 부분(기존 방식과 동일한 정의)
    ang = _math.radians(gen.overhang_angle_deg)
    step = det_h / _math.tan(ang)
    overhang = []
    for i in range(len(det_slices)):
        cur = clean(det_slices[i])
        if i == 0 or cur.is_empty:
            overhang.append(_Polygon())
            continue
        below = clean(det_slices[i - 1])
        overhang.append(cur.difference(below.buffer(step))
                        if not below.is_empty else cur)

    contacts = extract_contact_points(
        overhang, heights,
        max_area_per_point=contact_params.bead_diameter_mm ** 2 * 3,
    )
    if verbose:
        print(f"      접촉점 {len(contacts)}개")
    if not contacts:
        return SupportResult(trimesh.Trimesh(), None, det_slices)

    d_contact = contact_params.bead_diameter_mm
    d_body = body_params.bead_diameter_mm
    z_off = 0.5 * d_contact + gen.contact_z_gap_mm
    skeleton = grow_branches(
        contacts, det_slices, heights, gen,
        step_h=max(det_h * 3, d_contact * 0.5),
        merge_distance=gen.branch_merge_distance_mm or d_contact * 6,
        max_branch_angle_deg=gen.branch_angle_deg,
        contact_z_offset=z_off,
        bead_radius=0.5 * max(d_contact, d_body),
    )
    roots = skeleton.roots()
    n_bed = sum(1 for r in roots if skeleton.nodes[r].on_bed)
    n_model = sum(1 for r in roots if skeleton.nodes[r].on_model)
    if verbose:
        print(f"      골격: {skeleton.summary()}")
        print(f"      트렁크 {len(roots)}개 (베드 {n_bed} / 모델 윗면에 얹힘 {n_model})")
        if skeleton.skipped_contacts:
            print(f"      ! 베드에 너무 가까워 구슬이 못 들어가는 접촉점 "
                  f"{skeleton.skipped_contacts}개는 건너뜀")
        if skeleton.blocked_contacts:
            print(f"      ! 벽에 너무 붙어 구슬 둘 자리가 없는 접촉점 "
                  f"{skeleton.blocked_contacts}개는 건너뜀")

    if gen.adaptive_bead_size:
        assign_hierarchical_radii(
            skeleton, d_contact, d_body,
            max_trunk_diameter_mm=gen.tree_max_trunk_diameter_mm,
            slenderness=gen.tree_trunk_slenderness)

    seeds = skeleton_to_bead_seeds(
        skeleton, d_contact, d_body,
        model_slices=det_slices, heights=heights,
        xy_clearance=gen.xy_clearance_mm,
        max_branch_angle_deg=gen.branch_angle_deg,
        include_on_model=not gen.support_on_build_plate_only,
    )
    if gen.tree_bracing:
        braces, n_ties, n_braces = add_bracing(
            skeleton, d_body, det_slices, heights, gen.xy_clearance_mm,
            max_distance_mm=gen.tree_brace_distance_mm,
            brace_angle_deg=min(35.0, gen.overhang_angle_deg))
        seeds = seeds + braces
        if verbose:
            print(f"      트렁크 연결: 베드 연결 {n_ties}개, X 가새 {n_braces}쌍")
    seeds, n_drop = settle_collisions(
        seeds, mesh, gen.xy_clearance_mm, z_gap=gen.contact_z_gap_mm,
        model_slices=det_slices, heights=heights)
    seeds, n_float = prune_floating(seeds, mesh, z0, gen.xy_clearance_mm)
    n_drop += n_float
    if verbose:
        print(f"      구슬 {len(seeds)}개" +
              (f" (모델과 겹치거나 떨어져 나가 뺀 구슬 {n_drop}개)" if n_drop else ""))
        rep = structure_report(seeds, bed_z=z0)
        print(f"      구조 점검: 연결 덩어리 {rep['components']}개, "
              f"베드에 안 이어진 구슬 {rep['floating_beads']}개"
              f"(모델 윗면에 얹힌 가지 포함), "
              f"아래 받침 없는 구슬 {rep['unsupported_beads']}개")

    limit = gen.max_beads if gen.max_beads else bead_budget(detail)
    guard_bead_count(len(seeds), min(limit, bead_budget(detail)))

    # 구슬 좌표를 BeadPlan 형태로 담아 기존 meshing/검증 코드를 재사용한다.
    plan = BeadPlan()
    bead_h = gen.layer_height_mm
    by_layer: Dict[int, list] = {}
    for x, y, z, d in seeds:
        li = max(0, int((z - z0) / bead_h))
        by_layer.setdefault(li, []).append(
            {"x": x, "y": y, "angle": 0.0,
             "region": SupportBeadRegion.BODY, "d": d,
             "z_exact": z})
    for li in sorted(by_layer):
        plan.layers.append({
            "layer": li,
            "z_bottom": z0 + li * bead_h,
            "solid": None,
            "beads": by_layer[li],
        })

    support_mesh = plan_to_mesh(plan, gen, detail=detail)
    return SupportResult(support_mesh, plan, det_slices)


def generate_support(
    mesh: trimesh.Trimesh,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    detail: int = 1,
    verbose: bool = True,
) -> SupportResult:
    # --- 0) 검증: 조용히 틀린 결과를 내느니 명확한 에러로 막는다 -------------
    validate_mesh(mesh)
    validate_bead_params(contact_params, "인터페이스 구슬")
    validate_bead_params(body_params, "몸통 구슬")
    validate_gen_params(gen)
    check_model_scale(mesh, contact_params)
    if not 0 <= detail <= 3:
        raise InvalidParameterError(
            f"구 표면 세분화 단계가 {detail} 입니다. 0~3 사이여야 합니다."
        )

    # 슬라이싱 전에 몇 초 안에 끝나는 거친 사전 점검을 한다.
    #
    # 정밀 계산은 전체 슬라이싱 + 영역 계산을 거쳐야 하는데, 복잡한 모델
    # (나뭇가지 모양 거치대, 삼각형 190만 개)에서는 그것만으로 68초가
    # 걸렸다. 그동안 브라우저·프록시가 연결을 끊어서 서버가 멈춘 것처럼
    # 보였다. 확실히 과한 경우만 미리 걸러서, 몇 초 안에 원인과 해결법을
    # 알려준다.
    quick_limit = gen.max_beads if gen.max_beads else bead_budget(detail)
    quick = quick_overhang_estimate(mesh, gen, contact_params)
    guard_quick_estimate(quick, quick_limit)

    # --- 1) 탐지: 모델 형상을 제대로 볼 수 있는 얇은 층으로 자른다 -----------
    # 구슬 격자 간격(=layer_height_mm)으로 자르면, 굵은 펠릿을 쓸 때 층이
    # 듬성듬성해져서 그 사이의 오버행을 통째로 놓친다. 모델이 서포터를
    # 필요로 하는지는 모델 형상의 문제이지 펠릿 크기와 무관해야 한다.
    det_h = gen.detection_layer_height_mm or min(0.4, gen.layer_height_mm)
    det_h = max(det_h, float(mesh.extents[2]) / gen.max_detection_layers)
    det_slices, _ = slice_model(mesh, det_h, gen.max_detection_layers)
    if verbose:
        print(f"      탐지 슬라이싱: {len(det_slices)}층 @ {det_h:.3f}mm")

    # 구슬을 꺼낼 수 있는 통로 폭은 구슬 지름으로 본다.
    if not gen.allow_internal_supports:
        gen = replace(gen, removal_opening_mm=contact_params.bead_diameter_mm)
    support, contact = build_support_regions(det_slices, gen, det_h)
    if sum(1 for s in support if not clean(s).is_empty) == 0:
        return SupportResult(trimesh.Trimesh(), None, det_slices)

    # 구슬 수를 미리 어림해서, 메모리가 터져 멈추기 전에 막는다.
    # (탐지 층 기준이라 실제 구슬 층보다 촘촘해 안전한 쪽으로 과대평가된다)
    # 구슬이 영역에 비해 너무 크면 아무리 잘 깔아도 끊긴다. 미리 알려준다.
    fit_warning = check_bead_fits_regions(
        support, contact_params, gen.nozzle_diameter_mm)
    if fit_warning:
        warnings.warn(fit_warning, stacklevel=2)
        if verbose:
            print(f"      ! {fit_warning}")

    estimated = estimate_bead_count(support, gen, contact_params)
    # 상한은 구슬 면 수에 따라 달라진다. 사용자가 명시하면 그 값을 쓴다.
    limit = gen.max_beads if gen.max_beads else bead_budget(detail)

    # --- 나뭇가지(트리) 골격 모드 ------------------------------------------
    #
    # 기존 방식은 "오버행 영역을 격자로 최대한 채운다"였다. 트리 모드는
    # 먼저 골격(접촉점 -> 가지 -> 병합 -> 베드)을 정하고, 그 골격을 구슬로
    # 표현한다. 오버행 탐지와 공동(꺼낼 수 없는 곳) 판정은 위에서 이미 끝난
    # 것을 그대로 재사용하므로, 검증된 전처리는 건드리지 않는다.
    if gen.tree_enabled:
        result = _generate_tree_support(
            mesh, gen, contact_params, body_params,
            det_slices, det_h, z0=float(mesh.bounds[0][2]),
            detail=detail, verbose=verbose,
        )
        return result

    guard_bead_count(estimated, min(limit, bead_budget(detail)))

    # --- 2) 배치: 구슬 격자 간격으로 위 결과를 다시 샘플링한다 --------------
    z0 = float(mesh.bounds[0][2])
    bead_h = gen.layer_height_mm
    # 맨 아래 구슬은 베드에 얹혀야 한다. 구슬 중심을 층 중앙에 두면 반지름만큼
    # 베드 아래로 파고들어 슬라이서가 잘라내 버리므로, 첫 층 중심을 반지름
    # 높이에 맞추고 그 위로 격자 간격만큼 쌓는다.
    r0 = 0.5 * contact_params.bead_diameter_mm
    z_first = z0 + r0
    n_bead = max(1, int(math.ceil(float(mesh.extents[2]) / bead_h)))

    # 세분(필러) 구슬 지름들. 기본 구슬부터 인쇄 가능 최소까지 shrink 비율로
    # 내려간다. 아래에서 각 지름마다 '자기 크기에 맞는' 충돌 범위로 따로
    # 영역을 깎아 둔다 — 세대별로 나눠 두지 않으면, 기본(큰) 구슬 기준으로
    # 이미 통째로 비워진 층에는 정작 들어갈 수 있는 작은 구슬도 시도조차
    # 못 한다. 무한 큐브 꼭대기에서 실측: 기본구슬(r=1.25) 기준 영역 0mm^2,
    # 작은구슬(r=0.5) 기준 영역 772.6mm^2 — 같은 층인데 큰 구슬 기준으로만
    # 판정해서 위쪽이 통째로 끊겼다.
    min_bead = gen.min_bead_diameter_mm or (
        gen.nozzle_diameter_mm * gen.min_bead_to_nozzle_ratio)
    gen_diameters = [contact_params.bead_diameter_mm]
    d = contact_params.bead_diameter_mm
    for _ in range(gen.fill_generations):
        d *= gen.fill_shrink
        if d < min_bead:
            break
        gen_diameters.append(d)

    def collision_clip(region, z, bead_d):
        r = 0.5 * bead_d
        k_lo = min(len(det_slices) - 1, max(0, int((z - r - z0) / det_h)))
        k_hi = min(len(det_slices) - 1,
                   max(0, int((z + r + gen.contact_z_gap_mm - z0) / det_h)))
        if k_hi <= k_lo or clean(region).is_empty:
            return clean(region)
        spanned = [clean(det_slices[m]) for m in range(k_lo, k_hi + 1)]
        spanned = [g for g in spanned if not g.is_empty]
        if not spanned:
            return clean(region)
        blocked = unary_union(spanned).buffer(gen.xy_clearance_mm)
        return clean(clean(region).difference(blocked))

    bead_support, bead_contact = [], []
    # 세대별[층] 형태. 0번이 기본 구슬, 1번부터 세분 구슬(점점 작아짐).
    bead_support_by_gen = [[] for _ in gen_diameters]
    for j in range(n_bead):
        z = z_first + j * bead_h  # 그 층 구슬 중심의 높이
        k = min(len(det_slices) - 1, max(0, int((z - z0) / det_h)))
        region = support[k]

        # 구슬은 중심 층에만 있는 게 아니라 위아래로 반지름만큼 뻗는다.
        # 중심 층 하나만 보고 영역을 정하면, 그 사이 층에서 모델이 더 튀어나온
        # 경우 구슬이 모델을 파고든다(노즐 5mm 에서 구슬이 탐지 층 4.7개를
        # 걸쳐서, 실측 25~59개가 충돌했다).
        for gi, bead_d in enumerate(gen_diameters):
            clipped = collision_clip(region, z, bead_d)
            bead_support_by_gen[gi].append(clipped)
        bead_support.append(bead_support_by_gen[0][-1])
        bead_contact.append(contact[k])
    if verbose:
        n_used = sum(1 for s in bead_support if not clean(s).is_empty)
        print(f"      구슬 층 {n_bead}개 @ {bead_h:.3f}mm, 서포터 필요 {n_used}개")

    # 격자 원점은 오브젝트 전체에서 딱 한 번만 정해진다.
    # 격자 원점은 모델 '중심'에 맞춘다.
    #
    # 예전에는 bbox 의 최소 꼭짓점(좌측 끝)에 격자를 박아 두었다. 그러면 모델
    # 폭이 pitch 의 정수배가 아닐 때 좌우 격자가 근본적으로 어긋나서, 좌우
    # 대칭인 모델인데도 서포터가 비대칭으로 나온다(배 모델에서 거울상 일치율
    # 2%). 중심을 격자점으로 삼으면 좌우가 같은 위상으로 깔려 대칭이 된다.
    grid_origin = (
        0.5 * float(mesh.bounds[0][0] + mesh.bounds[1][0]),
        0.5 * float(mesh.bounds[0][1] + mesh.bounds[1][1]),
    )
    plan = plan_beads(
        bead_support, bead_contact, gen, contact_params, body_params,
        grid_origin, z_first - 0.5 * bead_h,  # meshing 이 +h/2 해서 중심을 잡는다
        raw_support=bead_support_by_gen,
    )

    # 공중에 뜬 구슬 덩어리는 지우기 전에 먼저 '이어 붙인다'.
    #
    # 서포터 영역은 바닥까지 이어져 있는데 중간 층의 영역이 너무 좁아 격자점이
    # 안 들어가면, 영역은 연속인데 구슬 사슬만 끊긴다. 그대로 지워 버리면
    # 정작 받쳐야 할 곳이 통째로 비어 서포터 역할을 못 한다. 아래로 기둥을
    # 내려 베드나 모델까지 연결하면 구조를 유지하면서 빈 구간만 메울 수 있다.
    if gen.stitch_floating:
        try:
            added = repair_connectivity(
                plan, gen, contact_params, det_slices, det_h,
                z0=float(mesh.bounds[0][2]), stride=gen.stitch_stride,
            )
            if verbose and added:
                print(f"      끊긴 구슬 사슬에 {added}개 이어 붙임")
        except Exception:
            pass  # scipy 가 없으면 이 복구 단계만 건너뛴다

    # 아래에 받쳐줄 것이 없는 구슬은 실제로 인쇄되지 않고 노즐에 끌려다닌다.
    if gen.prune_unsupported:
        try:
            dropped = prune_unsupported_beads(plan, gen, contact_params, mesh)
            if verbose and dropped:
                print(f"      공중에 뜬 구슬 {dropped}개 제거")
        except Exception:
            pass  # scipy 가 없으면 이 정리 단계만 건너뛴다

    # 같은 자리에 겹쳐 놓인 구슬은 그 지점만 과압출된다.
    try:
        dup = deduplicate_layer_beads(plan, contact_params.pitch_mm() * 0.5)
        if verbose and dup:
            print(f"      같은 자리에 겹친 구슬 {dup}개 정리")
    except Exception:
        pass

    # 이웃이 너무 적은 구슬은 힘을 전달하지 못하고 노즐에 걸려 떨어진다.
    if gen.min_bead_contacts > 0:
        try:
            weak = prune_weakly_connected(
                plan, gen, contact_params, z0=float(mesh.bounds[0][2]),
                min_contacts=gen.min_bead_contacts,
                det_slices=det_slices, det_h=det_h,
            )
            if verbose and weak:
                print(f"      이웃이 {gen.min_bead_contacts}개 미만인 구슬 {weak}개 제거")
        except Exception:
            pass

    # 아무것도 받치지 않는 구슬 뭉치는 재료만 쓰고 노즐에 걸리기만 한다.
    if gen.prune_orphan_clusters:
        try:
            orphaned = prune_orphan_clusters(
                plan, gen, contact_params, mesh, z0=float(mesh.bounds[0][2])
            )
            if verbose and orphaned:
                print(f"      아무것도 안 받치는 구슬 뭉치 {orphaned}개 제거")
        except Exception:
            pass

    # 위 정리 단계들이 구슬을 지우면, 그 위에 얹혀 있던 구슬이 새로 허공에
    # 뜬다. 허공 검사를 맨 처음 한 번만 하면 그렇게 생긴 구슬을 못 잡는다.
    # 더 지워지는 것이 없을 때까지 다시 훑는다.
    if gen.prune_unsupported:
        for _ in range(5):
            try:
                again = prune_unsupported_beads(plan, gen, contact_params, mesh)
            except Exception:
                break
            if not again:
                break
            if verbose:
                print(f"      정리 후 새로 뜬 구슬 {again}개 추가 제거")

    if verbose:
        print(f"      bead {sum(len(l['beads']) for l in plan.layers)}개")
    return SupportResult(plan_to_mesh(plan, gen, detail), plan, det_slices)
