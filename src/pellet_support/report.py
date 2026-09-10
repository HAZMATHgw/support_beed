# -*- coding: utf-8 -*-
"""리포트, 검증, 미리보기, 좌표 내보내기."""

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np
import trimesh

from .meshing import BeadPlan
from .params import SupportBeadParams, SupportBeadRegion, SupportGenParams
from .slicing import clean


def load_mesh(path: str) -> trimesh.Trimesh:
    try:
        obj = trimesh.load(path, force="mesh")
    except ModuleNotFoundError as exc:
        # trimesh 는 포맷별 로더의 의존성을 선택 설치로 둔다. 특히 3MF 는
        # 오브젝트 계층을 푸는 데 networkx 가 필요한데, 없으면 파일을 열지도
        # 못하고 여기서 죽는다. 원인을 그대로 노출하면 사용자가 헤매므로
        # 무엇을 설치하면 되는지 알려준다.
        # trimesh 는 예외를 재생성해서 넘기므로 exc.name 이 비어 있을 수 있다.
        # 그럴 땐 메시지에서 패키지 이름을 직접 뽑는다.
        missing = getattr(exc, "name", None)
        if not missing:
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
            missing = m.group(1).split(".")[0] if m else None
        if missing:
            raise RuntimeError(
                f"'{missing}' 패키지가 없어서 이 형식을 열 수 없습니다. "
                f"터미널에서 다음을 실행한 뒤 다시 시도하세요:  pip install {missing}"
            ) from exc
        raise RuntimeError(f"이 형식을 여는 데 필요한 패키지가 없습니다: {exc}") from exc

    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(list(obj.dump()))
    if not isinstance(obj, trimesh.Trimesh) or obj.is_empty:
        raise RuntimeError(f"메쉬를 읽지 못했습니다: {path}")
    return obj


def report_packing(contact: SupportBeadParams, body: SupportBeadParams) -> None:
    for name, p in (("contact", contact), ("body   ", body)):
        print(
            f"      [{name}] D={p.bead_diameter_mm:.3f}"
            f"  delta={p.lattice_overlap_ratio:.3f}"
            f"  접촉원지름={2 * p.contact_disc_radius_mm():.3f}mm"
            f"  배위수={p.coordination_number()}"
            f"  충전율={p.packing_fraction():.2f}"
            f"  mm3/mm={p.mm3_per_mm():.3f}"
        )
        if p.lattice_overlap_ratio > 0.12:
            print(f"      ! [{name.strip()}] delta 가 큽니다. 구가 뭉쳐 분리가 어려워집니다.")
        elif p.lattice_overlap_ratio <= 0.0:
            print(f"      ! [{name.strip()}] 구가 서로 닿지 않아 고정력이 없습니다.")


def measure_packing(plan: BeadPlan, gen: SupportGenParams, target_pitch: float,
                    tol: float = 0.02, vertical_pitch: Optional[float] = None):
    """실제로 찍힌 좌표에서 최근접 이웃 거리와 배위수를 측정한다.

    설계값이 아니라 결과물을 재는 것이므로, 격자 원점 고정과 시프트 누적이
    의도대로 동작했는지 확인하는 용도로 쓴다.
    """
    from scipy.spatial import cKDTree

    # 격자가 의도대로 만들어졌는지 보는 검사이므로 '기본 구슬'만 센다.
    # 좁은 곳을 메운 세분 구슬과 사슬을 이어 붙인 구슬은 일부러 격자를
    # 벗어나 있어서, 함께 세면 배위수가 0 으로 나온다.
    pts = []
    for layer in plan.layers:
        z_center = layer["z_bottom"] + 0.5 * gen.layer_height_mm
        pts.extend(
            (b["x"], b["y"], z_center)
            for b in layer["beads"]
            if not b.get("refined") and not b.get("stitch")
        )
    if len(pts) < 2:
        return None

    arr = np.asarray(pts)
    tree = cKDTree(arr)
    # k 의 첫 번째는 자기 자신(거리 0)이라 잘라낸다.
    dist, _ = tree.query(arr, k=min(13, len(arr)))
    dist = dist[:, 1:]
    # 비등방 충전에서는 가로 이웃과 세로 이웃의 거리가 다르다.
    # 한쪽만 세면 배위수가 0 으로 나온다.
    hit = np.abs(dist - target_pitch) < tol
    if vertical_pitch and abs(vertical_pitch - target_pitch) > tol:
        hit = hit | (np.abs(dist - vertical_pitch) < tol)
    touching = hit.sum(axis=1)
    return {
        "median_nearest": float(np.median(dist[:, 0])),
        "median_coordination": int(np.median(touching)),
        "n_beads": len(arr),
    }


def verify_packing(plan: BeadPlan, gen: SupportGenParams,
                   contact: SupportBeadParams) -> None:
    target = contact.pitch_mm()
    vertical = contact.vertical_neighbor_distance_mm()
    stats = measure_packing(plan, gen, target, vertical_pitch=vertical)
    if stats is None:
        return
    print(
        f"      검증: 최근접거리 중앙값 {stats['median_nearest']:.3f} mm "
        f"(목표 {target:.3f}), 배위수 중앙값 {stats['median_coordination']}"
    )


def export_preview(plan: BeadPlan, slices, layer_id: int, path: str) -> None:
    """해당 층(채움)과 바로 위 층(점선)을 겹쳐 그려 hollow 안착을 확인."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.patches import Polygon as MplPoly

    fig, ax = plt.subplots(figsize=(8, 6))

    def draw(geom, **kw):
        if geom is None or geom.is_empty:
            return
        for p in geom.geoms if hasattr(geom, "geoms") else [geom]:
            if p.is_empty or not hasattr(p, "exterior"):
                continue
            ax.add_patch(MplPoly(np.array(p.exterior.coords), closed=True, **kw))

    draw(clean(slices[layer_id]), facecolor="#d8d8d8", edgecolor="#909090", lw=0.6)
    entry = plan.layers[layer_id]
    if entry["solid"] is not None:
        draw(entry["solid"], facecolor="#9ec5ff", edgecolor="none", alpha=0.7)
    for b in entry["beads"]:
        color = "#ff7043" if b["region"] == SupportBeadRegion.CONTACT else "#3f6fd8"
        ax.add_patch(
            Circle((b["x"], b["y"]), 0.5 * b["d"], facecolor=color,
                   edgecolor="white", lw=0.3, alpha=0.9)
        )
    nxt = layer_id + 1
    if nxt < len(plan.layers):
        for b in plan.layers[nxt]["beads"]:
            ax.add_patch(
                Circle((b["x"], b["y"]), 0.5 * b["d"], facecolor="none",
                       edgecolor="#111111", lw=0.7, ls="--", alpha=0.6)
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(
        f"layer {layer_id}  z={entry['z_bottom']:.2f}mm  "
        f"beads={len(entry['beads'])}   (dashed = next layer, nested in hollows)"
    )
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_plan_json(plan: BeadPlan, path: str) -> None:
    """C++ 구현과 좌표 단위로 대조할 때 쓰는 덤프."""
    data = [
        {
            "layer": l["layer"],
            "z_bottom": round(l["z_bottom"], 4),
            "solid": l["solid"] is not None,
            "beads": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                for b in l["beads"]
            ],
        }
        for l in plan.layers
        if l["beads"] or l["solid"] is not None
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
