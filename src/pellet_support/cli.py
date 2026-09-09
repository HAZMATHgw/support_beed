# -*- coding: utf-8 -*-
"""커맨드라인 인터페이스."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

from .params import SupportGenParams
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
        description="3D 모델 -> 최밀충전 구형 펠릿 서포터",
    )
    ap.add_argument("input", help="입력 모델 (stl/obj/3mf/ply/off)")
    ap.add_argument("-o", "--output", default=None)

    g = ap.add_argument_group("충전 기하")
    g.add_argument("--nozzle", type=float, default=1.0,
                   help="노즐 지름 (mm). 압출 가능한 최소 크기의 기준")
    g.add_argument("--bead-diameter", type=float, default=None,
                   help="구슬 지름 (mm). 비우면 노즐 지름의 절반. "
                        "노즐보다 작게 잡을수록 gap/interface 사각지대가 줄어들지만 "
                        "구슬 개수(=인쇄 시간)는 늘어난다")
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
    s.add_argument("--build-plate-only", action="store_true")
    s.add_argument("--allow-internal-supports", action="store_true",
                   help="모델 내부의 닫힌 공동에도 서포터를 채움 "
                        "(기본은 꺼짐 — 출력 후 꺼낼 수 없으므로)")
    s.add_argument("--max-layers", type=int, default=4000)

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


def _run(argv=None) -> int:
    args = build_parser().parse_args(argv)

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
        max_layers=args.max_layers,
    )

    print(f"[1/4] 모델 로드: {args.input}")
    mesh = load_mesh(args.input)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    print(f"      삼각형 {len(mesh.faces)}개, 크기 {np.round(mesh.extents, 2)} mm")

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
        verify_packing(result.plan, gen, contact_params)
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
        out2 = f"{root}_with_support{ext}"
        scene.export(out2)
        print(f"      합본 저장: {out2}")
    if args.preview is not None and result.plan is not None:
        li = args.preview
        if li < 0:
            cand = [l["layer"] for l in result.plan.layers if l["beads"]]
            li = cand[-1] if cand else 0
        li = max(0, min(li, len(result.plan.layers) - 1))
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
