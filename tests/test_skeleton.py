# -*- coding: utf-8 -*-
"""나뭇가지(트리) 골격 방식 검증.

analysis 단계에서 합의한 검증 기준을 그대로 테스트로 옮긴다.

    구조: Tree topology ~= Bead topology
    접촉: Tree contact points ~= Bead contact points
    연결: 모든 서포터가 베드 또는 유효한 체인에 연결
    충돌: Model collision 최소화
    효율: 불필요한 Bead 최소화
    안정성: 작은 branch -> 큰 trunk 계층 유지
"""

import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from pellet_support import SupportGenParams, generate_support, make_params
from pellet_support.slicing import clean, slice_model
from pellet_support.skeleton import (
    assign_hierarchical_radii,
    extract_contact_points,
    fix_residual_collisions,
    grow_branches,
    skeleton_to_bead_seeds,
)


def _table_mesh():
    leg = trimesh.creation.box(extents=[6, 6, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[24, 24, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def _run_skeleton(mesh, nozzle=1.0, hierarchical=True):
    contact, body = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    det_h = min(0.4, gen.layer_height_mm)
    sl, heights = slice_model(mesh, det_h, gen.max_detection_layers)
    ang = math.radians(gen.overhang_angle_deg)
    step = det_h / math.tan(ang)
    overhang = []
    for i in range(len(sl)):
        cur = clean(sl[i])
        if i == 0 or cur.is_empty:
            overhang.append(Polygon())
            continue
        below = clean(sl[i - 1])
        overhang.append(cur.difference(below.buffer(step))
                        if not below.is_empty else cur)
    pts = extract_contact_points(
        overhang, heights, max_area_per_point=contact.bead_diameter_mm ** 2 * 3)
    z_off = 0.5 * contact.bead_diameter_mm + gen.contact_z_gap_mm
    sk = grow_branches(
        pts, sl, heights, gen,
        step_h=max(det_h * 3, contact.bead_diameter_mm * 0.5),
        merge_distance=contact.bead_diameter_mm * 6, contact_z_offset=z_off)
    if hierarchical:
        assign_hierarchical_radii(sk, contact.bead_diameter_mm, body.bead_diameter_mm)
    seeds = skeleton_to_bead_seeds(
        sk, contact.bead_diameter_mm, body.bead_diameter_mm,
        model_slices=sl, heights=heights, xy_clearance=gen.xy_clearance_mm)
    seeds = fix_residual_collisions(seeds, mesh, gen.xy_clearance_mm)
    return sk, seeds, pts, gen, contact


# --- Test 1: 단순 오버행 -----------------------------------------------

def test_single_overhang_produces_one_branch_to_bed():
    """다리 하나 위에 상판 하나: 접촉점 여러 개가 결국 가지 몇 개로 정리
    되고, 전부 베드까지 이어져야 한다."""
    mesh = _table_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh)
    assert len(pts) > 1, "테스트 모델 자체가 접촉점을 여러 개 내야 의미가 있다"
    assert len(sk.roots()) < len(pts), "가지가 하나도 안 합쳐졌다"
    assert len(seeds) > 0
    zmin = min(s[2] for s in seeds)
    assert zmin < 1.0, "가장 낮은 구슬이 베드 근처까지 안 내려왔다"


# --- Test 2: 여러 접촉점이 하나의 트리로 합쳐지는지 ------------------------

def test_multiple_contact_points_merge_into_fewer_trunks():
    mesh = _table_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh)
    assert len(sk.roots()) <= len(pts) // 2, (
        f"접촉점 {len(pts)}개가 트렁크 {len(sk.roots())}개로밖에 안 줄었다")


# --- Test 3: 계층 구조(트렁크 굵게, 말단 가늘게) --------------------------

def test_trunk_is_thicker_than_terminal_branches():
    mesh = _table_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh, hierarchical=True)
    contact_radii = [n.radius for n in sk.nodes if n.kind == "contact"]
    trunk_radii = [n.radius for n, i in zip(sk.nodes, range(len(sk.nodes)))
                  if n.parent is None]
    assert trunk_radii, "트렁크(루트) 노드가 없다"
    assert max(trunk_radii) >= max(contact_radii), (
        "트렁크가 말단 접촉점보다 가늘다 — 계층 구조가 반대로 됐다")


# --- Test 4: 무한 큐브(기존 문제 재현 모델) --------------------------------

def test_infinity_cube_top_connection_and_coverage(tmp_path):
    """예전에 '맨 위쪽 연결 없음' 문제가 있었던 모델. 커버리지와 연결성을
    격자 방식과 비교한다."""
    cube_path = "/tmp/new_uploads/cube.3mf"
    try:
        mesh = trimesh.load(cube_path, force="mesh")
    except Exception:
        pytest.skip("무한 큐브 예제 파일이 없어 건너뜀")
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    sk, seeds, pts, gen, contact = _run_skeleton(mesh, nozzle=5.0)
    assert seeds, "구슬이 하나도 안 나왔다"
    pq = trimesh.proximity.ProximityQuery(mesh)
    P = np.array([(x, y, z) for x, y, z, d in seeds])
    D = np.array([d for x, y, z, d in seeds])
    dist = np.abs(pq.signed_distance(P))
    collide = int((dist < 0.5 * D - 0.02).sum())
    assert collide == 0, f"모델 관통 {collide}개"

    # 기존 격자 방식보다 구슬 수가 적어야 한다(핵심 목표)
    from pellet_support import generate_support
    old = generate_support(mesh, gen, contact,
                           make_params(nozzle_diameter_mm=5.0)[1],
                           detail=0, verbose=False)
    n_old = sum(len(l["beads"]) for l in old.plan.layers) if old.plan else 0
    assert len(seeds) < n_old, (
        f"트리 방식({len(seeds)})이 격자 방식({n_old})보다 많다")


# --- Test 5: 물리적 유효성(관통 없음) --------------------------------------

def test_no_model_penetration_across_nozzle_sizes():
    mesh = _table_mesh()
    for nozzle in (1.0, 3.0):
        sk, seeds, pts, gen, contact = _run_skeleton(mesh, nozzle=nozzle)
        pq = trimesh.proximity.ProximityQuery(mesh)
        P = np.array([(x, y, z) for x, y, z, d in seeds])
        D = np.array([d for x, y, z, d in seeds])
        dist = np.abs(pq.signed_distance(P))
        collide = int((dist < 0.5 * D - 0.02).sum())
        assert collide == 0, f"노즐 {nozzle}mm 에서 관통 {collide}개"


# --- Test 6: 파이프라인 통합 (tree_enabled 스위치) --------------------------

def test_tree_mode_is_opt_in_and_default_is_unaffected():
    """tree_enabled 기본값(False)이면 기존 격자 방식 그대로여야 한다."""
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    assert gen.tree_enabled is False


def test_tree_mode_runs_through_the_real_pipeline():
    mesh = _table_mesh()
    contact, body = make_params(nozzle_diameter_mm=3.0)
    gen = SupportGenParams(nozzle_diameter_mm=3.0,
                           layer_height_mm=contact.layer_height_mm(),
                           tree_enabled=True)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert result.plan is not None
    n = sum(len(l["beads"]) for l in result.plan.layers)
    assert n > 0
    assert not result.mesh.is_empty


def test_tree_mode_uses_fewer_beads_than_grid_mode():
    mesh = _table_mesh()
    contact, body = make_params(nozzle_diameter_mm=3.0)
    gen_tree = SupportGenParams(nozzle_diameter_mm=3.0,
                                layer_height_mm=contact.layer_height_mm(),
                                tree_enabled=True)
    gen_grid = SupportGenParams(nozzle_diameter_mm=3.0,
                                layer_height_mm=contact.layer_height_mm(),
                                tree_enabled=False)
    r_tree = generate_support(mesh, gen_tree, contact, body, detail=0, verbose=False)
    r_grid = generate_support(mesh, gen_grid, contact, body, detail=0, verbose=False)
    n_tree = sum(len(l["beads"]) for l in r_tree.plan.layers)
    n_grid = sum(len(l["beads"]) for l in r_grid.plan.layers)
    assert n_tree < n_grid, f"트리({n_tree}) >= 격자({n_grid})"


def test_no_branch_gets_stuck_on_an_obstacle():
    """막힌 가지는 옆으로 우회해서 반드시 베드까지 내려가야 한다.

    예전에는 막히면 그 층에서 대기만 했다. 그러면 가지가 장애물 위에 갇혀
    병합도 못 하고 트렁크만 늘어난다(실측: 배 모델에서 트렁크 7개 중 3개가
    z=9.60, 1.05, 1.05 에서 멈춰 있었다).

    우회 반경을 기울임 한계(max_lean)에 묶어두면 안 된다는 것도 여기서
    드러났다 — 선체 벽을 넘으려면 수 mm 가 필요한데 0.58mm 만 훑고
    포기했었다.
    """
    mesh = _table_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh)
    bed_z = min(s[2] for s in seeds)
    stuck = [i for i in sk.roots() if sk.nodes[i].z > bed_z + 2.0]
    assert not stuck, (
        f"트렁크 {len(stuck)}개가 베드에 못 닿고 도중에 멈췄다: "
        f"{[round(sk.nodes[i].z, 2) for i in stuck]}")


def test_detour_radius_is_not_limited_by_lean_angle():
    """우회 탐색은 기울임 한계보다 훨씬 멀리까지 볼 수 있어야 한다."""
    from shapely.geometry import box as shapely_box

    from pellet_support.skeleton import _find_detour

    # 폭 10mm 짜리 벽 한가운데에서 시작 -> 최소 5mm 는 가야 벗어난다
    wall = shapely_box(-5, -50, 5, 50)
    found = _find_detour(0.0, 0.0, wall, xy_clearance=0.5,
                         max_step=0.5)  # 기울임 한계는 0.5mm 로 아주 작게
    assert found is not None, "우회로를 못 찾았다"
    assert abs(found[0]) > 5.0, (
        f"벽(폭 10mm)을 못 벗어났다: {found}")


def test_precision_pass_survives_a_large_mesh():
    """거대 메쉬에서도 정밀 관통 보정이 죽지 않아야 한다.

    trimesh 의 ProximityQuery.on_surface 는 **삼각형 수 x 조회 점 수**에
    비례해 메모리를 쓴다. 처음엔 삼각형 수만 문제인 줄 알았는데(192만
    각형에서 죽고 22만은 정상), 10만 각형으로 줄여도 2,583점을 한 번에
    넣으면 죽었고 22만 각형이라도 2,000점씩 끊으면 멀쩡했다.
    단순화 + 청크 처리 둘 다 필요하다.
    """
    from pellet_support.skeleton import fix_residual_collisions

    # 삼각형을 충분히 많이 만든다(세분화된 구)
    mesh = trimesh.creation.icosphere(subdivisions=7, radius=20)
    mesh.apply_translation([0, 0, 25])
    assert len(mesh.faces) > 300_000, "테스트 전제: 큰 메쉬여야 의미가 있다"

    seeds = [(float(x), 0.0, 5.0, 2.0) for x in np.linspace(-40, 40, 1500)]
    out = fix_residual_collisions(seeds, mesh, xy_clearance=0.5)
    assert len(out) == len(seeds), "보정 후 구슬 수가 달라졌다"


def test_precision_pass_actually_pushes_beads_out():
    """보정이 실제로 관통을 없애는지(그냥 통과시키는 게 아닌지) 확인."""
    from pellet_support.skeleton import fix_residual_collisions

    mesh = trimesh.creation.box(extents=[20, 20, 20])
    mesh.apply_translation([0, 0, 10])
    # 상자 한가운데를 관통하는 구슬
    seeds = [(0.0, 0.0, 10.0, 4.0)]
    out = fix_residual_collisions(seeds, mesh, xy_clearance=0.5)
    x, y, z, d = out[0]
    moved = math.hypot(x, y) + abs(z - 10.0)
    assert moved > 0.1, "관통 구슬이 전혀 안 움직였다"


# --- 구조물로서의 유효성(출력 가능 + 서로 이어짐) ---------------------------
#
# 예전 구현은 병합을 '같은 높이에서 옆으로' 해서 공중에 뜬 수평 구슬 사슬이
# 생겼고(실측: 엣지 35~43% 가 사실상 수평, 구슬 14~35% 가 받침 없이 허공에
# 찍힘), 트렁크가 구슬 1개짜리 외줄이라 옆 지지가 없었다. 아래 테스트는
# 좌표·골격을 직접 재서 그 상태로 돌아가지 않는지 본다.

def _cantilever_mesh(height=40.0, reach=30.0):
    """베드에서 올라온 기둥 끝에 옆으로 뻗은 두꺼운 팔 두 개(양쪽).

    팔 아랫면 전체가 높은 오버행이라 긴 트렁크가 필요하다.
    """
    post = trimesh.creation.box(extents=[6, 6, height])
    post.apply_translation([0, 0, height / 2])
    arm = trimesh.creation.box(extents=[2 * reach, 8, 4])
    arm.apply_translation([0, 0, height + 2])
    mesh = trimesh.util.concatenate([post, arm])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def test_branches_never_move_sideways_faster_than_printable():
    """모든 가지 구간은 45° 이내여야 한다(수평 병합 다리 금지)."""
    mesh = _table_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh)
    worst = 0.0
    for child, node in enumerate(sk.nodes):
        if node.parent is None:
            continue
        par = sk.nodes[node.parent]
        dz = abs(node.z - par.z)
        h = math.hypot(node.x - par.x, node.y - par.y)
        if dz < 1e-9:
            assert h < 1e-6, f"수평 엣지가 있다(길이 {h:.2f}mm)"
            continue
        worst = max(worst, math.degrees(math.atan2(h, dz)))
    assert worst <= 45.5, f"가장 누운 가지가 수직에서 {worst:.1f}° 기울었다"


def test_tree_output_is_printable_and_grounded():
    """출력 좌표에서 직접 잰다: 받침 없이 허공에 찍히는 구슬과 베드에
    안 이어진 구슬이 거의 없어야 한다."""
    from pellet_support.skeleton import structure_report

    mesh = _cantilever_mesh()
    contact, body = make_params(nozzle_diameter_mm=3.0)
    gen = SupportGenParams(nozzle_diameter_mm=3.0,
                           layer_height_mm=contact.layer_height_mm(),
                           tree_enabled=True)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    seeds = [(b["x"], b["y"], b["z_exact"], b["d"])
             for l in result.plan.layers for b in l["beads"]]
    rep = structure_report(seeds, bed_z=0.0)
    assert rep["unsupported_beads"] <= 0.08 * rep["beads"], rep
    assert rep["floating_beads"] <= 0.02 * rep["beads"], rep


def test_tall_trunk_gets_wider_toward_the_bed():
    """높이 40mm 오버행을 받치는 트렁크 밑동은 구슬 1개보다 훨씬 굵어야 한다."""
    from pellet_support.skeleton import structure_report  # noqa: F401

    mesh = _cantilever_mesh()
    sk, seeds, pts, gen, contact = _run_skeleton(mesh, nozzle=3.0)
    body_d = make_params(nozzle_diameter_mm=3.0)[1].bead_diameter_mm
    tall = [r for r in sk.roots() if sk.nodes[r].on_bed]
    assert tall
    widest = max(2 * sk.nodes[r].radius for r in tall)
    assert widest >= 40.0 / gen.tree_trunk_slenderness * 0.9, (
        f"밑동 굵기 {widest:.2f}mm — 세장비 {gen.tree_trunk_slenderness} 기준 미달")
    assert widest > 2.5 * body_d


def test_bracing_ties_separate_trunks_together():
    """양쪽 팔 아래 트렁크들이 따로 서 있지 않고 하나로 이어져야 한다."""
    from pellet_support.skeleton import structure_report

    mesh = _cantilever_mesh()
    contact, body = make_params(nozzle_diameter_mm=3.0)
    base = dict(nozzle_diameter_mm=3.0, layer_height_mm=contact.layer_height_mm(),
                tree_enabled=True)

    def comps(**kw):
        gen = SupportGenParams(**base, **kw)
        r = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
        seeds = [(b["x"], b["y"], b["z_exact"], b["d"])
                 for l in r.plan.layers for b in l["beads"]]
        return structure_report(seeds, bed_z=0.0)["components"]

    with_brace = comps(tree_bracing=True, tree_brace_distance_mm=80.0)
    without = comps(tree_bracing=False)
    assert with_brace < without, (with_brace, without)
    assert with_brace <= 2, f"가새를 넣어도 덩어리 {with_brace}개로 흩어져 있다"


def test_contact_beads_are_not_deleted_by_collision_check():
    """오버행 바로 밑 접촉 구슬은 z 간격 기준으로 봐야 한다 — XY 여유로
    재면 접촉 구슬이 전부 '관통'으로 잡혀 빠진다."""
    from pellet_support.skeleton import settle_collisions

    mesh = trimesh.creation.box(extents=[20, 20, 4])
    mesh.apply_translation([0, 0, 12])  # 아랫면 z=10
    seeds = [(float(x), 0.0, 10.0 - 0.5 - 0.3, 1.0) for x in range(-5, 6)]
    out, dropped = settle_collisions(seeds, mesh, xy_clearance=0.8, z_gap=0.3)
    assert dropped == 0
    assert len(out) == len(seeds)
