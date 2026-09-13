# -*- coding: utf-8 -*-
"""커맨드라인 인터페이스."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import trimesh

from .params import SupportGenParams
from dataclasses import replace

from .autotune import auto_tune_bead_diameter
from .pipeline import generate_support, make_params
from .validation import (
    InvalidModelError,
    InvalidParameterError,
    TooManyBeadsError,
)
from .report import (
    export_plan_json,
    export_preview,
    load_mesh,
    report_packing,
    verify_packing,
)

MESH_EXTS = (".stl", ".obj", ".3mf", ".ply", ".off")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pellet-support",
        description="3D 모델 -> 접점과 가지를 합치는 트리형 구슬 서포터 (기본)",
    )
    ap.add_argument("input", help="입력 모델 (stl/obj/3mf/ply/off)")
    ap.add_argument("-o", "--output", default=None)

    g = ap.add_argument_group("충전 기하")
    g.add_argument("--nozzle", type=float, default=0.4,
                   help="노즐 지름 (mm). 압출 가능한 최소 크기의 기준")
    g.add_argument("--bead-diameter", type=float, default=None,
                   help="구슬 지름 (mm). 비우면 노즐 지름의 절반. "
                        "노즐보다 작게 잡을수록 gap/interface 사각지대가 줄어들지만 "
                        "구슬 개수(=인쇄 시간)는 늘어난다")
    g.add_argument("--auto", action="store_true",
                   help="모델과 노즐에 맞는 구슬 크기를 자동으로 고른다. "
                        "--bead-diameter 를 직접 주면 그 값이 우선한다")
    g.add_argument("--auto-target", type=float, default=0.75,
                   help="--auto 의 목표 품질(서포터 영역 중 구슬이 들어갈 수 있는 "
                        "비율). 낮추면 얇고 좁은 오버행 구간이 통째로 '자리 없음' "
                        "으로 빠져 그 부분에 구슬이 하나도 안 놓일 수 있다 — "
                        "구슬을 더 키우고 싶으면 이 값보다 auto-tree-height-ratio "
                        "/ auto-void-ratio 를 올리는 쪽을 먼저 시도하세요")
    g.add_argument("--auto-void-ratio", type=float, default=0.35,
                   help="--auto --no-tree(격자 모드) 가 구슬 지름 상한을 정할 때 "
                        "쓰는 비율. 노즐 지름이 아니라 실제 오버행(빈 공간) 단면의 "
                        "평균 지름에 이 비율을 곱해 상한으로 쓴다")
    g.add_argument("--auto-tree-height-ratio", type=float, default=0.1,
                   help="--auto 트리 모드(기본)가 구슬 지름 상한을 정할 때 쓰는 "
                        "비율. 트리는 접점에서 베드까지 기둥을 세우므로, 오버행 "
                        "단면의 폭이 아니라 베드까지의 (면적 가중) 평균 높이에 이 "
                        "비율을 곱해 상한으로 쓴다. 배 선체처럼 단면은 얇아도 "
                        "높이가 큰 모델에서 구슬을 더 키워 개수를 줄이려면 올리세요")
    g.add_argument("--min-bead-ratio", type=float, default=None,
                   help="인쇄 가능한 최소 구슬 지름 / 노즐 지름 (기본 0.35). "
                        "압출기가 더 작은 방울을 안정적으로 뽑을 수 있으면 낮추세요")
    g.add_argument("--overlap", type=float, default=None,
                   help="delta. 이웃과 눌리는 정도 (권장 0.04~0.08)")
    g.add_argument("--lateral-overlap", type=float, default=None,
                   help="가로(평면) 방향 겹침. 좌우 흔들림에 버티는 힘을 결정한다. "
                        "--vertical-overlap 과 함께 쓰면 비등방 충전이 된다 "
                        "(예: 가로 0.14 / 세로 0.04)")
    g.add_argument("--vertical-overlap", type=float, default=None,
                   help="세로(층 사이) 방향 겹침. 작을수록 펠릿으로 잘 부서진다")
    g.add_argument("--body-bead-ratio", type=float, default=0.97,
                   help="몸통 구 지름 / contact 구 지름. 낮출수록 잘 부서짐")
    g.add_argument("--stagger-period", type=int, default=3,
                   help="2=ABAB(hcp), 3=ABCABC(fcc)")
    g.add_argument("--straight-columns", action="store_true",
                   help="구형 충전을 끄고 수직 기둥으로 (배위수 8)")
    g.add_argument("--segment-ratio", type=float, default=None)
    g.add_argument("--edge-margin-ratio", type=float, default=None)

    s = ap.add_argument_group("서포터 영역")
    s.add_argument("--layer-height", type=float, default=None,
                   help="비우면 충전 기하에서 자동 계산 (권장)")
    s.add_argument("--overhang-angle", type=float, default=45.0)
    s.add_argument("--z-gap-layers", type=int, default=1)
    s.add_argument("--xy-clearance", type=float, default=0.8)
    s.add_argument("--contact-layers", type=int, default=2)
    s.add_argument("--solid-first-layers", type=int, default=1)
    s.add_argument("--min-island-area", type=float, default=2.0)
    s.add_argument("--no-fallback-solid", action="store_true",
                   help="구슬로 못 채운 좁은 자리를 일반 서포터(얇은 벽)로 "
                        "메우지 않는다")
    s.add_argument("--interactive", "-i", action="store_true",
                   help="노즐 지름 · 구슬 지름을 명령줄 대신 터미널에서 "
                        "직접 물어본다. 다른 옵션은 그대로 --옵션 으로 줄 수 있다")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--tree", dest="tree", action="store_true",
                      help="나뭇가지(트리) 골격 방식 (기본). 접점을 성기게 골라 "
                           "가지를 합치며 내려가 구슬 수를 줄인다")
    mode.add_argument("--no-tree", "--grid", dest="tree", action="store_false",
                      help="트리 대신 기존 격자 충전 방식으로 서포터 영역을 채운다")
    ap.set_defaults(tree=True)
    s.add_argument("--tree-contact-spacing", type=float, default=None, metavar="MM",
                   help="트리 접점 간격 (mm). 비우면 구슬 지름의 4배. "
                        "키울수록 접점과 구슬 수가 줄어든다")
    s.add_argument("--branch-angle", type=float, default=25.0, metavar="DEG",
                   help="트리 가지의 최대 기울기 (수직 기준, 기본 25도)")
    s.add_argument("--branch-merge-distance", type=float, default=None, metavar="MM",
                   help="트리 가지를 병합할 최대 수평 거리 (mm). "
                        "비우면 구슬 지름의 6배")
    s.add_argument("--trunk-slenderness", type=float, default=8.0,
                   help="트리 모드 트렁크 세장비 상한(높이/굵기). 트렁크가 아래로 "
                        "갈수록 이 비율로 굵어지는 구슬 다발이 된다. 작을수록 튼튼하고 "
                        "구슬이 많이 든다(기본 8)")
    s.add_argument("--no-bracing", action="store_true",
                   help="트리 모드에서 이웃 트렁크끼리 베드 연결 + X 가새로 잇지 않는다")
    s.add_argument("--brace-distance", type=float, default=None,
                   help="트리 모드에서 서로 이을 트렁크 사이 최대 거리(mm). "
                        "비우면 몸통 구슬 지름의 20배")
    s.add_argument("--build-plate-only", action="store_true")
    s.add_argument("--allow-internal-supports", action="store_true",
                   help="모델 내부의 닫힌 공동에도 서포터를 채움 "
                        "(기본은 꺼짐 — 출력 후 꺼낼 수 없으므로)")
    s.add_argument("--max-layers", type=int, default=4000)
    s.add_argument("--max-beads", type=int, default=None,
                   help="구슬 수 상한. 비우면 메모리에서 자동 계산")

    o = ap.add_argument_group("출력")
    o.add_argument("--sphere-detail", type=int, default=1,
                   help="0=20면, 1=80면, 2=320면")
    o.add_argument("--with-model", action="store_true",
                   help="모델+서포터 합본도 저장")
    o.add_argument("--dump-json", action="store_true",
                   help="bead 좌표를 JSON 으로도 저장")
    o.add_argument("--preview", type=int, default=None, metavar="LAYER",
                   help="해당 층 배치를 PNG 로 저장 (-1 = 최상단)")
    return ap


def _prompt_float(label: str, default: Optional[float] = None,
                  allow_empty: bool = False) -> Optional[float]:
    """터미널에서 숫자 하나를 물어본다. 잘못 입력하면 다시 물어본다."""
    suffix = ""
    if default is not None:
        suffix = f" [기본 {default}]"
    elif allow_empty:
        suffix = " [비우면 자동]"
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return default
            if allow_empty:
                return None
            print("  값을 입력해야 합니다.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print(f"  숫자로 입력해 주세요(입력값: '{raw}').")
            continue
        if value <= 0:
            print("  0보다 큰 값이어야 합니다.")
            continue
        return value


def _prompt_bool(label: str, default: bool = False) -> bool:
    """터미널에서 예/아니오를 물어본다."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "ㅇ", "예"):
            return True
        if raw in ("n", "no", "ㄴ", "아니오"):
            return False
        print("  y 또는 n 으로 답해 주세요.")


def _run_interactive_prompts(args) -> None:
    """--interactive 일 때 노즐·구슬 지름을 터미널에서 직접 물어본다.

    명령줄에 --nozzle/--bead-diameter 를 이미 줬다면 그 값을 기본값으로
    보여주고, 그냥 Enter 치면 그 값을 그대로 쓴다.
    """
    print("=== 대화형 입력 ===")
    args.nozzle = _prompt_float(
        "노즐 지름(mm) — 압출기에 실제로 꽂힌 노즐 크기",
        default=args.nozzle)
    args.bead_diameter = _prompt_float(
        "구슬 지름(mm) — 비우면 노즐의 절반으로 자동",
        default=args.bead_diameter, allow_empty=True)
    args.tree = _prompt_bool(
        "나뭇가지(트리) 골격 방식을 쓸까요? — 접점과 구슬 수를 줄임",
        default=args.tree)
    if not args.with_model:
        args.with_model = _prompt_bool(
            "모델과 서포터를 합친 파일도 함께 받을까요?",
            default=True)
    print("===================\n")


def _run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.interactive:
        _run_interactive_prompts(args)

    contact_params, body_params = make_params(
        nozzle_diameter_mm=args.nozzle,
        bead_diameter_mm=args.bead_diameter,
        overlap=args.overlap,
        lateral_overlap=args.lateral_overlap,
        vertical_overlap=args.vertical_overlap,
        body_bead_ratio=args.body_bead_ratio,
        stagger_period=args.stagger_period,
        straight_columns=args.straight_columns,
        segment_ratio=args.segment_ratio,
        edge_margin_ratio=args.edge_margin_ratio,
    )

    layer_h = args.layer_height or contact_params.layer_height_mm()
    gen = SupportGenParams(
        nozzle_diameter_mm=args.nozzle,
        layer_height_mm=layer_h,
        overhang_angle_deg=args.overhang_angle,
        contact_z_gap_layers=args.z_gap_layers,
        xy_clearance_mm=args.xy_clearance,
        contact_layers=args.contact_layers,
        solid_first_layers=args.solid_first_layers,
        min_island_area_mm2=args.min_island_area,
        min_bead_to_nozzle_ratio=(args.min_bead_ratio
                                  if args.min_bead_ratio is not None else 0.35),
        support_on_build_plate_only=args.build_plate_only,
        allow_internal_supports=args.allow_internal_supports,
        tree_enabled=args.tree,
        tree_contact_spacing_mm=args.tree_contact_spacing,
        branch_angle_deg=args.branch_angle,
        branch_merge_distance_mm=args.branch_merge_distance,
        tree_trunk_slenderness=args.trunk_slenderness,
        tree_bracing=not args.no_bracing,
        tree_brace_distance_mm=args.brace_distance,
        fallback_solid=not args.no_fallback_solid,
        max_layers=args.max_layers,
        max_beads=args.max_beads,
        auto_void_bead_ratio=args.auto_void_ratio,
        auto_tree_height_ratio=args.auto_tree_height_ratio,
    )

    print(f"[1/4] 모델 로드: {args.input}")
    mesh = load_mesh(args.input)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    print(f"      삼각형 {len(mesh.faces)}개, 크기 {np.round(mesh.extents, 2)} mm")

    # 자동 튜닝: 모델과 노즐을 보고 구슬 크기를 고른다.
    # 사용자가 --bead-diameter 를 직접 준 경우에는 그 값을 존중한다.
    if args.auto and args.bead_diameter is None:
        tuned = auto_tune_bead_diameter(
            mesh, gen, contact_params, target_fill=args.auto_target,
            max_beads=args.max_beads,
        )
        print("[1.5/4] 자동 튜닝")
        print("      " + tuned.summary().replace("\n", "\n      "))
        contact_params, body_params = make_params(
            nozzle_diameter_mm=args.nozzle,
            bead_diameter_mm=tuned.chosen.bead_diameter_mm,
            overlap=args.overlap,
            lateral_overlap=args.lateral_overlap,
            vertical_overlap=args.vertical_overlap,
            body_bead_ratio=args.body_bead_ratio,
            stagger_period=args.stagger_period,
            straight_columns=args.straight_columns,
            segment_ratio=args.segment_ratio,
            edge_margin_ratio=args.edge_margin_ratio,
        )
        layer_h = contact_params.layer_height_mm()
        gen = replace(gen, layer_height_mm=layer_h)

    print("[2/4] 충전 기하")
    print(f"      pitch = {contact_params.pitch_mm():.3f} mm,  "
          f"층높이 = {layer_h:.3f} mm  <- 프로파일에 이 값을 쓰세요")
    report_packing(contact_params, body_params)
    if args.layer_height is not None:
        req = contact_params.layer_height_mm()
        if abs(args.layer_height - req) > 0.02:
            print(f"      ! 층높이가 {req:.3f} mm 여야 구가 아래층과 닿습니다.")

    print("[3/4] 오버행 탐색 + bead 배치")
    result = generate_support(
        mesh, gen, contact_params, body_params, args.sphere_detail
    )
    if result.mesh.is_empty:
        print("      서포터가 필요 없습니다.")
        return 0
    print(f"      삼각형 {len(result.mesh.faces)}개, "
          f"부피 약 {abs(result.mesh.volume) / 1000.0:.2f} cm^3")
    try:
        if not gen.tree_enabled:
            verify_packing(result.plan, gen, contact_params)
        else:
            n = sum(len(l["beads"]) for l in result.plan.layers)
            print(f"      검증: 트리 골격 구슬 {n}개(격자 지표는 트리 방식에는 "
                  f"적용되지 않아 생략)")
    except Exception:
        pass  # scipy 가 없으면 검증만 건너뛴다

    root, ext = os.path.splitext(args.input)
    ext = ext.lower() if ext.lower() in MESH_EXTS else ".stl"
    out = args.output or f"{root}_support{ext}"
    result.mesh.export(out)
    print(f"[4/4] 저장: {out}")

    if args.with_model:
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="model")
        scene.add_geometry(result.mesh, node_name="support")
        # -o 로 출력 위치를 지정했으면 합본 파일도 그 위치를 따라야 한다.
        # 예전에는 -o 와 무관하게 항상 '입력 파일' 옆에 만들어서, -o 로
        # 다른 폴더(예: /tmp)를 지정했을 때 합본 파일만 엉뚱한 곳에
        # 생겨 사용자가 못 찾는 문제가 있었다.
        out_root, out_ext_ = os.path.splitext(out)
        out2 = f"{out_root}_with_model{out_ext_ or ext}"
        scene.export(out2)
        print(f"      합본 저장: {out2}")
    if args.preview is not None and result.plan is not None:
        li = args.preview
        if li < 0:
            cand = [l["layer"] for l in result.plan.layers if l["beads"]]
            li = cand[-1] if cand else 0
        # 트리 계획은 빈 층을 생략하므로 층 번호와 배열 인덱스가 다르다.
        li = min((layer["layer"] for layer in result.plan.layers), key=lambda n: abs(n - li))
        out3 = f"{root}_layer{li}.png"
        export_preview(result.plan, result.slices, li, out3)
        print(f"      미리보기 저장: {out3}")
    if args.dump_json and result.plan is not None:
        out4 = f"{root}_beads.json"
        export_plan_json(result.plan, out4)
        print(f"      bead 좌표 저장: {out4}")
    return 0


def main(argv=None) -> int:
    """사용자에게는 traceback 대신 무엇을 고쳐야 하는지만 보여준다."""
    try:
        return _run(argv)
    except (InvalidParameterError, InvalidModelError) as exc:
        print(f"\n입력값 오류: {exc}", file=sys.stderr)
        return 2
    except TooManyBeadsError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n중단했습니다.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        # 의존성 누락 등, 이미 사람이 읽을 수 있게 포장된 오류
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
