# -*- coding: utf-8 -*-
"""층별 배치 계획을 만들고, 그것을 실제 구 메쉬로 바꾼다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from shapely.geometry import Point

from .lattice import bead_angle, support_bead_generate_centers
from .packing import refine_thin_region
from .params import SupportBeadParams, SupportBeadRegion, SupportGenParams
from .slicing import clean


@dataclass
class BeadPlan:
    """층별 bead 배치 결과. 메쉬 생성과 좌표 내보내기가 공용으로 쓴다."""

    layers: List[dict] = field(default_factory=list)


def plan_beads(
    support,
    contact,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    grid_origin: Tuple[float, float],
    z0: float,
) -> BeadPlan:
    plan = BeadPlan()
    # 인쇄 가능한 최소 구슬 지름. 노즐보다 지나치게 작은 구슬은 압출기가
    # 안정적으로 뽑지 못하므로 그 아래로는 세분하지 않는다.
    min_bead = gen.min_bead_diameter_mm or (
        gen.nozzle_diameter_mm * gen.min_bead_to_nozzle_ratio)
    for i, sup in enumerate(support):
        sup = clean(sup)
        entry = {
            "layer": i,
            "z_bottom": z0 + i * gen.layer_height_mm,
            "solid": None,
            "beads": [],
        }
        if sup.is_empty:
            plan.layers.append(entry)
            continue

        # 첫 층은 접착/구조 때문에 원래대로 꽉 채운다.
        if i < gen.solid_first_layers:
            entry["solid"] = sup
            plan.layers.append(entry)
            continue

        con = clean(contact[i]).intersection(sup)
        body = sup.difference(con) if not con.is_empty else sup
        for region, geom, prm in (
            (SupportBeadRegion.CONTACT, con, contact_params),
            (SupportBeadRegion.BODY, body, body_params),
        ):
            geom = clean(geom)
            if geom.is_empty:
                continue
            ang = bead_angle(prm, i)
            placed = support_bead_generate_centers(geom, prm, i, grid_origin)
            for c in placed:
                entry["beads"].append(
                    {
                        "x": c.x,
                        "y": c.y,
                        "angle": ang,
                        "region": region,
                        "d": prm.bead_diameter_mm,
                    }
                )

            # 기본 구슬이 하나도 안 들어간 조각은 더 작은 구슬로 다시 채운다.
            #
            # 폭이 구슬보다 좁은 조각은 격자점이 통째로 비껴가서 예전에는
            # 아무것도 안 놓였고, 그 층에서 구슬 사슬이 끊겼다. 3D 프린터가
            # 층을 쌓듯 큰 구슬로 먼저 채우고 남은 곳을 작은 구슬로 메운다.
            if gen.refine_thin_regions and min_bead > 0:
                parts = geom.geoms if hasattr(geom, "geoms") else [geom]
                covered = [Point(c.x, c.y) for c in placed]
                for part in parts:
                    if part.is_empty:
                        continue
                    if any(part.contains(pt) for pt in covered):
                        continue  # 이 조각은 이미 기본 구슬이 채웠다
                    fine, fine_d = refine_thin_region(
                        part, prm, i, grid_origin, min_bead
                    )
                    for c in fine:
                        entry["beads"].append({
                            "x": c.x, "y": c.y, "angle": ang,
                            "region": region, "d": fine_d, "refined": True,
                        })
        plan.layers.append(entry)
    return plan


def _unit_sphere(detail: int):
    sphere = trimesh.creation.icosphere(subdivisions=max(0, detail), radius=1.0)
    return np.asarray(sphere.vertices), np.asarray(sphere.faces)


def plan_to_mesh(
    plan: BeadPlan, gen: SupportGenParams, detail: int = 1
) -> trimesh.Trimesh:
    """bead 를 실제 구로 찍는다.

    수만 개를 하나씩 만들면 느리므로 단위구를 한 번만 만들고 반지름이 같은
    구들끼리 묶어 넘파이 브로드캐스팅으로 한 번에 평행이동한다. 겹치는
    구들은 슬라이서가 합집합으로 처리한다.
    """
    unit_v, unit_f = _unit_sphere(detail)
    h = gen.layer_height_mm

    by_radius: Dict[float, List[Tuple[float, float, float]]] = {}
    parts: List[trimesh.Trimesh] = []

    for entry in plan.layers:
        z_center = entry["z_bottom"] + 0.5 * h
        if entry["solid"] is not None:
            geom = entry["solid"]
            for poly in geom.geoms if hasattr(geom, "geoms") else [geom]:
                if poly.is_empty or poly.area < 1e-9:
                    continue
                try:
                    m = trimesh.creation.extrude_polygon(poly, h)
                except ValueError as exc:
                    # trimesh 는 폴리곤을 삼각형으로 쪼갤 엔진을 선택 설치로 둔다.
                    # 없으면 "No available triangulation engine!" 만 던지고 죽는데,
                    # 그 메시지만으로는 무엇을 깔아야 할지 알 수 없다.
                    if "triangulation engine" not in str(exc):
                        raise
                    raise RuntimeError(
                        "폴리곤을 삼각형으로 쪼갤 엔진이 없습니다(첫 층 바닥 생성에 필요). "
                        "터미널에서 다음을 실행하세요:  pip install mapbox-earcut"
                    ) from exc
                m.apply_translation([0, 0, entry["z_bottom"]])
                parts.append(m)
        for b in entry["beads"]:
            by_radius.setdefault(round(0.5 * b["d"], 5), []).append(
                (b["x"], b["y"], z_center)
            )

    for radius, pts in by_radius.items():
        centers = np.asarray(pts, dtype=np.float64)
        verts = (unit_v[None, :, :] * radius + centers[:, None, :]).reshape(-1, 3)
        offset = (np.arange(len(centers)) * len(unit_v))[:, None, None]
        faces = (unit_f[None, :, :] + offset).reshape(-1, 3)
        parts.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)
