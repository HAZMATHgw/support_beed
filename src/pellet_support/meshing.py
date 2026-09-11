# -*- coding: utf-8 -*-
"""층별 배치 계획을 만들고, 그것을 실제 구 메쉬로 바꾼다."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from shapely.geometry import Point
from shapely.ops import unary_union

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
    raw_support=None,
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
        # 기본(0번 세대) 구슬 영역이 비어도, 더 작은 세분 구슬이 들어갈
        # 자리가 있으면 그냥 넘어가면 안 된다. 무한 큐브 꼭대기에서 정확히
        # 이 경우(기본 0mm^2, 세분 772.6mm^2)가 있었는데, 이 조기 종료
        # 하나 때문에 아래쪽의 세분화 코드까지 통째로 도달을 못 했다.
        any_filler = raw_support and any(
            i < len(gens) and not clean(gens[i]).is_empty
            for gi, gens in enumerate(raw_support) if gi > 0
        )
        if sup.is_empty and not any_filler:
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
            ang = bead_angle(prm, i)
            placed = []
            if not geom.is_empty:
                # 충돌 조건에서 직접 유도한 여백(필요 이상으로 깎지 않는다).
                #
                # 여백을 딱 (반지름 - 여유)로 두면 남는 간극이 정확히 0이라
                # 폴리곤 근사·부동소수점 오차만큼 모델을 파고든다(실측 0.004mm).
                # 구슬 지름의 5%를 안전 여유로 더한다.
                safety = 0.05 * prm.bead_diameter_mm
                margin = max(
                    0.0, 0.5 * prm.bead_diameter_mm - gen.xy_clearance_mm + safety)
                margin = min(margin, prm.bead_diameter_mm * prm.edge_margin_ratio)
                placed = support_bead_generate_centers(
                    geom, prm, i, grid_origin, edge_margin_override=margin)
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
            elif not gen.refine_thin_regions or not raw_support:
                # 기본 구슬 영역도 비었고 세분화도 안 쓰면 정말로 할 게 없다.
                # (geom 이 비었다고 무조건 건너뛰면 안 된다 — 기본 구슬 기준
                # 으로만 통째로 빈 층에서 작은 구슬은 들어갈 수 있는 경우가
                # 있다. 무한 큐브 꼭대기에서 기본구슬 0mm^2, 세분구슬(한 단계
                # 작은) 772.6mm^2 였던 층이 바로 이 경우였는데, 예전에는
                # 여기서 그냥 건너뛰어서 세분화 코드 자체가 실행되지 않았다.)
                continue

            # 남은 빈 곳을 더 작은 구슬로 메운다(다중 크기 충전).
            #
            # 예전에는 '구슬이 하나도 안 들어간 조각'만 다시 시도했다. 그래서
            # 큰 구슬이 몇 개라도 들어간 영역은 나머지가 비어도 그대로 뒀고,
            # 경사면 아래처럼 영역이 좁아지는 곳에 큰 틈이 남았다.
            # 이제는 실제로 덮인 면적을 빼고 남은 곳을 지름을 줄여가며 채운다.
            if gen.refine_thin_regions and min_bead > 0:
                leftover = geom
                covered_parts = []
                if placed:
                    footprints = unary_union([
                        Point(c.x, c.y).buffer(0.5 * prm.bead_diameter_mm, 4)
                        for c in placed
                    ])
                    leftover = clean(geom.difference(footprints))
                    covered_parts.append(footprints)

                # 세대별 원본 영역(자기 크기에 맞는 충돌창으로 깎인 것).
                # geom(=세대 0, 기본 구슬 기준)만 갖고 세분화를 하면, 기본
                # 구슬 기준으로 이미 통째로 비워진 층에서는 leftover 도
                # 처음부터 비어 있어 작은 구슬을 시도조차 못 한다. 무한
                # 큐브 꼭대기에서 기본구슬(r=1.25) 기준 0mm^2, 작은구슬
                # (r=0.5) 기준 772.6mm^2 였던 게 이 문제였다.
                # contact[i] 는 collision 클리핑 이전의 '인터페이스냐 몸통이냐'
                # 순수 분류 경계라서, 이걸로 각 세대의 permissive 영역을
                # 다시 갈라야 gen-0 의 좁은 경계에 갇히지 않는다.
                fine_d = prm.bead_diameter_mm
                for gidx in range(1, gen.fill_generations + 1):
                    fine_d *= gen.fill_shrink
                    if fine_d < min_bead:
                        break
                    candidate = leftover
                    if raw_support and gidx < len(raw_support):
                        raw_whole = clean(raw_support[gidx][i])
                        if not raw_whole.is_empty:
                            con_g = clean(contact[i]).intersection(raw_whole)
                            this_gen_region = (
                                con_g if region == SupportBeadRegion.CONTACT
                                else clean(raw_whole.difference(con_g))
                            )
                            covered = (unary_union(covered_parts)
                                      if covered_parts else None)
                            candidate = (
                                clean(this_gen_region.difference(covered))
                                if covered else this_gen_region
                            )
                    if candidate.is_empty or candidate.area < 1e-6:
                        continue
                    fine_prm = replace(prm, bead_diameter_mm=fine_d)
                    fine_margin = max(
                        0.0, 0.5 * fine_d - gen.xy_clearance_mm + 0.05 * fine_d)
                    fine = support_bead_generate_centers(
                        candidate, fine_prm, i, grid_origin,
                        edge_margin_override=fine_margin,
                    )
                    if not fine:
                        continue
                    for c in fine:
                        entry["beads"].append({
                            "x": c.x, "y": c.y, "angle": ang,
                            "region": region, "d": fine_d, "refined": True,
                        })
                    new_footprints = unary_union([
                        Point(c.x, c.y).buffer(0.5 * fine_d, 4) for c in fine
                    ])
                    covered_parts.append(new_footprints)
                    leftover = clean(leftover.difference(new_footprints))

                # 구슬로는 끝내 못 채운 부분은 일반 서포터(얇은 벽)로 메운다.
                #
                # 여기 남은 조각은 인쇄 가능한 최소 구슬보다도 좁은 곳이다.
                # 그대로 비워 두면 그 층에서 받침이 끊기므로, 폴리곤을 그대로
                # 압출해 얇은 벽으로 세운다. 구슬처럼 낱알로 부서지지는 않지만
                # 원래 얇아서 손으로 뜯어내기 쉽다.
                if gen.fallback_solid and not leftover.is_empty:
                    keep = []
                    for p in (leftover.geoms
                              if hasattr(leftover, "geoms") else [leftover]):
                        if p.is_empty or p.area < gen.fallback_min_area_mm2:
                            continue
                        # 곡면을 따라 잘린 조각이라 꼭짓점이 수천 개씩 된다
                        # (실측 중앙값 1,145개, 최대 20,562개). 그대로 압출하면
                        # 삼각형이 213배로 폭증한다. 형상에 영향이 거의 없는
                        # 수준으로 단순화한다.
                        tol = 0.05 * prm.bead_diameter_mm
                        simple = clean(p.simplify(tol, preserve_topology=True))
                        if not simple.is_empty and simple.area >= gen.fallback_min_area_mm2:
                            keep.append(simple)
                    if keep:
                        entry.setdefault("solid_extra", []).extend(keep)
        plan.layers.append(entry)
    return plan


def _unit_sphere(detail: int):
    sphere = trimesh.creation.icosphere(subdivisions=max(0, detail), radius=1.0)
    return np.asarray(sphere.vertices), np.asarray(sphere.faces)


def deduplicate_layer_beads(plan, min_separation_mm: float) -> int:
    """같은 층에서 지나치게 가까운 구슬을 하나만 남긴다.

    사슬을 이어 붙이거나 좁은 곳을 세분해 채우다 보면 이미 구슬이 있는
    자리에 또 놓일 수 있다. 겹쳐 놓인 구슬은 압출량만 두 배로 쓰고 형상은
    그대로라, 그 자리만 과압출되어 노즐이 긁고 지나간다.
    """
    if min_separation_mm <= 0:
        return 0
    from scipy.spatial import cKDTree

    removed = 0
    for layer in plan.layers:
        beads = layer["beads"]
        if len(beads) < 2:
            continue
        xy = [(b["x"], b["y"]) for b in beads]
        pairs = cKDTree(xy).query_pairs(min_separation_mm, output_type="ndarray")
        if len(pairs) == 0:
            continue
        # 큰 구슬을 남기고 작은 쪽을 지운다
        drop = set()
        for i, j in pairs:
            if i in drop or j in drop:
                continue
            drop.add(j if beads[j]["d"] <= beads[i]["d"] else i)
        for idx in sorted(drop, reverse=True):
            del beads[idx]
        removed += len(drop)
    return removed


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
        # 첫 층 통판 + 구슬로 못 채운 자리의 얇은 벽을 함께 압출한다.
        solids = []
        if entry["solid"] is not None:
            geom = entry["solid"]
            solids.extend(geom.geoms if hasattr(geom, "geoms") else [geom])
        solids.extend(entry.get("solid_extra", []))
        if solids:
            for poly in solids:
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
            # 격자 방식 구슬은 층 중심에 놓이지만, 나뭇가지 골격에서 나온
            # 구슬은 가지 경로를 따라 임의 높이에 놓인다. z_exact 가 있으면
            # 그 값을 그대로 써야 한다(층 중심으로 스냅하면 가지 모양이
            # 계단처럼 뭉개지고, 애써 맞춘 관통 회피도 어긋난다).
            bz = b.get("z_exact", z_center)
            by_radius.setdefault(round(0.5 * b["d"], 5), []).append(
                (b["x"], b["y"], bz)
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
