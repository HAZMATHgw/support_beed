# -*- coding: utf-8 -*-
"""모델 입력부터 서포터 메쉬 출력까지 통째로 검증."""

import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point

from pellet_support import (
    SupportGenParams,
    generate_support,
    make_params,
    measure_packing,
    slice_model,
)
from pellet_support.slicing import clean


@pytest.fixture(scope="module")
def table_mesh():
    """다리 4개 위에 상판이 얹힌 모델. 상판 아랫면이 오버행이다."""
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = trimesh.creation.box(extents=[5, 5, 12])
            leg.apply_translation([sx * 10, sy * 6, 6])
            parts.append(leg)
    top = trimesh.creation.box(extents=[30, 20, 4])
    top.apply_translation([0, 0, 14])
    parts.append(top)
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


@pytest.fixture(scope="module")
def built(table_mesh):
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(
        nozzle_diameter_mm=1.0, layer_height_mm=contact.layer_height_mm()
    )
    result = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
    return result, gen, contact, body


def test_support_is_generated(built):
    result, _, _, _ = built
    assert not result.mesh.is_empty
    assert result.plan is not None
    assert sum(len(l["beads"]) for l in result.plan.layers) > 100


def test_measured_coordination_is_twelve(built):
    """설계값이 아니라 실제로 찍힌 좌표에서 배위수를 측정한다."""
    result, gen, contact, _ = built
    stats = measure_packing(result.plan, gen, contact.pitch_mm())
    assert stats is not None
    assert stats["median_nearest"] == pytest.approx(contact.pitch_mm(), abs=1e-3)
    assert stats["median_coordination"] == 12


def test_straight_columns_give_eight_neighbours(table_mesh):
    contact, body = make_params(nozzle_diameter_mm=1.0, straight_columns=True)
    gen = SupportGenParams(
        nozzle_diameter_mm=1.0, layer_height_mm=contact.layer_height_mm()
    )
    result = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
    stats = measure_packing(result.plan, gen, contact.pitch_mm())
    assert stats["median_coordination"] == 8


def test_beads_never_collide_with_the_model(built, table_mesh):
    """모든 bead 가 그 층 모델 단면에서 떨어져 있어야 한다."""
    result, gen, contact, body = built
    slices, _ = slice_model(table_mesh, gen.layer_height_mm, gen.max_layers)

    worst = math.inf
    for layer in result.plan.layers:
        geom = clean(slices[layer["layer"]])
        if geom.is_empty:
            continue
        for b in layer["beads"]:
            foot = Point(b["x"], b["y"]).buffer(0.5 * b["d"], 8)
            assert not foot.intersects(geom)
            worst = min(worst, geom.distance(foot))
    assert worst > 0.0


def test_support_stays_below_the_overhang(built, table_mesh):
    """서포터는 상판 아랫면(z=12)보다 아래에만 있어야 한다."""
    result, _, _, _ = built
    assert result.mesh.bounds[1][2] < 12.0
    # 맨 아래 구슬이 베드를 뚫고 내려가면 안 된다
    assert result.mesh.bounds[0][2] >= -1e-6


def test_no_support_for_a_model_without_overhangs():
    """단순 정육면체는 서포터가 필요 없다."""
    cube = trimesh.creation.box(extents=[10, 10, 10])
    cube.apply_translation([0, 0, 5])
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(
        nozzle_diameter_mm=1.0, layer_height_mm=contact.layer_height_mm()
    )
    result = generate_support(cube, gen, contact, body, detail=0, verbose=False)
    assert result.mesh.is_empty


def test_each_bead_is_a_closed_solid(built):
    """구 하나하나는 닫힌 메쉬여야 슬라이서가 합집합을 낼 수 있다."""
    result, _, _, _ = built
    comps = result.mesh.split(only_watertight=False)
    assert len(comps) > 10
    assert all(c.is_watertight for c in comps[:50])


def test_deterministic(table_mesh):
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(
        nozzle_diameter_mm=1.0, layer_height_mm=contact.layer_height_mm()
    )
    counts = []
    for _ in range(2):
        r = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
        counts.append(sum(len(l["beads"]) for l in r.plan.layers))
    assert counts[0] == counts[1]


def test_missing_optional_dependency_gives_actionable_message(tmp_path, monkeypatch):
    """trimesh 의 선택 의존성이 없을 때 무엇을 설치할지 알려준다."""
    import trimesh as _tm

    from pellet_support import report

    def boom(*a, **k):
        raise ModuleNotFoundError("No module named 'networkx'")

    monkeypatch.setattr(report.trimesh, "load", boom)
    path = tmp_path / "x.3mf"
    path.write_bytes(b"dummy")

    with pytest.raises(RuntimeError) as ei:
        report.load_mesh(str(path))
    assert "networkx" in str(ei.value)
    assert "pip install networkx" in str(ei.value)


def test_slicing_reports_missing_scipy_instead_of_returning_empty(monkeypatch):
    """의존성이 없을 때 '단면 없음'으로 둔갑시키지 말고 원인을 알려야 한다.

    이 실패를 조용히 삼키면 오버행이 하나도 안 잡혀서
    '이 모델에는 서포터가 필요하지 않습니다' 라는 거짓 결과가 나온다.
    """
    import numpy as np

    from pellet_support import slicing

    def boom(*a, **k):
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr(slicing, "edges_to_polygons", boom)
    segments = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]],
                         [[1, 1], [0, 1]], [[0, 1], [0, 0]]], dtype=float)

    with pytest.raises(RuntimeError) as ei:
        slicing.segments_to_polygons(segments)
    assert "scipy" in str(ei.value)
    assert "pip install scipy" in str(ei.value)


def test_missing_triangulation_engine_is_explained(monkeypatch, table_mesh):
    """삼각분할 엔진이 없을 때 무엇을 설치할지 알려준다."""
    from pellet_support import meshing

    def boom(*a, **k):
        raise ValueError("No available triangulation engine!")

    monkeypatch.setattr(meshing.trimesh.creation, "extrude_polygon", boom)
    contact, body = make_params(nozzle_diameter_mm=1.0)
    # 통판 첫 층을 켜야 extrude_polygon 경로를 탄다(기본값은 0 = 전부 구슬).
    gen = SupportGenParams(layer_height_mm=contact.layer_height_mm(),
                           solid_first_layers=1)

    with pytest.raises(RuntimeError) as ei:
        generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
    assert "mapbox-earcut" in str(ei.value)


def test_solid_first_layer_does_not_orphan_the_bead_layer_above_it(table_mesh):
    """통판 첫 층 위에 앉은 첫 구슬 층이 '떠 있다'고 오판되면 안 된다.

    solid_first_layers 로 첫 층을 beads 없는 통판으로 깔면, 연결성 검사가
    _bead_points 만 보고 그 층에서 점을 하나도 못 뽑는다. 스티칭
    (stitch_floating)이 그 구멍을 우연히 메워 주지만, 그게 꺼져 있거나
    (실패하거나 scipy 가 없어서) 실행되지 않으면 prune_unsupported_beads 가
    통판 바로 위 첫 구슬 층 전체를 '아래에 아무것도 없다'로 지우고, 그 위로
    전체가 연쇄적으로 무너진다(실측: 구슬 10만개 이상 전부 삭제).
    """
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(
        nozzle_diameter_mm=1.0,
        layer_height_mm=contact.layer_height_mm(),
        solid_first_layers=1,
        stitch_floating=False,
    )
    result = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
    assert result.plan is not None
    solid_layer, first_bead_layer = result.plan.layers[0], result.plan.layers[1]
    assert solid_layer["solid"] is not None
    assert len(first_bead_layer["beads"]) > 100
    assert sum(len(l["beads"]) for l in result.plan.layers) > 1000


def test_3mf_export_does_not_require_manually_installing_extra_packages():
    """trimesh[easy] 번들 하나로 3MF 왕복(로드+저장)이 전부 되는지 확인.

    networkx/scipy/mapbox-earcut/lxml/rtree 를 하나씩 쫓아다니며 개별
    지정하는 대신 trimesh 의 'easy' extra 에 의존하기로 한 결정에 대한
    회귀 테스트 — 이 테스트가 깨지면 다시 개별 패키지를 쫓아야 한다는 뜻이다.
    """
    import trimesh

    box = trimesh.creation.box(extents=[10, 10, 10])
    box.export("/tmp/_pellet_test_roundtrip.3mf")
    loaded = trimesh.load("/tmp/_pellet_test_roundtrip.3mf", force="mesh")
    assert not loaded.is_empty


def _bead_count(mesh, nozzle):
    contact, body = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    r = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    if r.plan is None:
        return 0, r
    return sum(len(l["beads"]) for l in r.plan.layers), r


def test_support_volume_is_stable_across_bead_sizes(table_mesh):
    """구슬을 키워도 '지지해야 할 부피'는 비슷해야 한다.

    예전에는 오버행 탐지 해상도가 구슬 격자 간격에 묶여 있어서, 굵은 펠릿을
    고르면 층이 듬성듬성해지고 그 사이 오버행을 통째로 놓쳐
    '서포터가 거의 필요 없다'는 결론이 나왔다.
    """
    vols = {}
    for nozzle in (1.0, 2.0, 3.0):
        _, r = _bead_count(table_mesh, nozzle)
        assert not r.mesh.is_empty, f"노즐 {nozzle}mm 에서 서포터가 사라졌다"
        vols[nozzle] = abs(r.mesh.volume)

    # 굵은 구슬이라고 서포터 부피가 한 자릿수로 쪼그라들면 안 된다
    assert min(vols.values()) > 0.25 * max(vols.values()), vols


def test_support_columns_reach_the_build_plate(table_mesh):
    """서포터는 모델 표면에 껍질처럼 붙지 말고 바닥까지 내려와야 한다.

    xy_clearance 로 잘린 영역을 누적값에까지 되먹임하면 완만한 경사면의
    얇은 오버행 링이 즉시 죽어서 기둥이 만들어지지 않는다.
    """
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(layer_height_mm=contact.layer_height_mm())
    r = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)

    zs = [l["z_bottom"] for l in r.plan.layers if l["beads"]]
    assert zs, "구슬이 하나도 없다"
    # 가장 낮은 구슬 층이 바닥 근처(한 층 이내)에 있어야 한다
    assert min(zs) < gen.layer_height_mm, min(zs)


def test_bottom_beads_sit_on_the_plate_not_below_it(table_mesh):
    """맨 아래 구슬이 베드 아래로 파고들면 슬라이서가 잘라낸다."""
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(layer_height_mm=contact.layer_height_mm())
    r = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)
    assert r.mesh.bounds[0][2] >= -1e-6, r.mesh.bounds[0][2]


def test_detection_resolution_is_independent_of_bead_lattice(table_mesh):
    """탐지 슬라이싱은 구슬 층높이가 아니라 자체 해상도를 쓴다."""
    from pellet_support.slicing import slice_model

    contact, body = make_params(nozzle_diameter_mm=4.0)
    gen = SupportGenParams(nozzle_diameter_mm=4.0,
                           layer_height_mm=contact.layer_height_mm())
    r = generate_support(table_mesh, gen, contact, body, detail=0, verbose=False)

    # 반환되는 slices 는 탐지용이므로, 구슬 층수보다 훨씬 촘촘해야 한다
    n_bead_layers = len(r.plan.layers)
    assert len(r.slices) > n_bead_layers * 3, (len(r.slices), n_bead_layers)


def _sealed_hollow_box():
    """사방이 막힌 속 빈 상자. 내부 공동은 출력 후 손이 닿지 않는다."""
    outer = trimesh.creation.box(extents=[30, 30, 30])
    inner = trimesh.creation.box(extents=[24, 24, 24])
    mesh = trimesh.util.concatenate([outer, inner])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def _run(mesh, nozzle=1.0, **gen_kw):
    contact, body = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm(), **gen_kw)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    if result.plan is None:
        return 0
    return sum(len(l["beads"]) for l in result.plan.layers)


def _beads_inside_cavity(mesh, half_extent=12.0, **gen_kw):
    """닫힌 공동(±half_extent) 안에 놓인 구슬 수를 센다.

    바깥쪽 브림/플레어는 정상 구조물이므로 전체 개수로 세면 안 된다.
    """
    import numpy as np

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm(), **gen_kw)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    if result.plan is None:
        return 0
    pts = [(b["x"], b["y"]) for l in result.plan.layers for b in l["beads"]]
    if not pts:
        return 0
    arr = np.asarray(pts)
    inside = (np.abs(arr[:, 0]) < half_extent) & (np.abs(arr[:, 1]) < half_extent)
    return int(inside.sum())


def test_no_support_inside_a_sealed_cavity():
    """닫힌 공동은 채우지 않는다.

    예전에는 속 빈 상자의 서포터 18,000여 개가 100% 내부에 갇혀 나왔다.
    꺼낼 수 없으므로 재료와 시간만 버리는 결과였다.
    모델 바깥쪽 브림/플레어는 흔들림을 잡는 정상 구조라 여기서 세지 않는다.
    """
    assert _beads_inside_cavity(_sealed_hollow_box()) == 0


def test_internal_supports_can_be_re_enabled():
    """필요하면 옵션으로 되살릴 수 있어야 한다."""
    assert _beads_inside_cavity(
        _sealed_hollow_box(), allow_internal_supports=True) > 0


def test_open_cavities_still_get_support(table_mesh):
    """위가 트인 공간(아치 아래)까지 지워버리면 안 된다."""
    assert _run(table_mesh) > 100


def test_tunnel_under_a_roof_still_gets_support():
    """양옆이 트이고 위만 막힌 터널도 서포터가 필요하다."""
    left = trimesh.creation.box(extents=[6, 30, 15])
    right = trimesh.creation.box(extents=[6, 30, 15])
    right.apply_translation([20, 0, 0])
    roof = trimesh.creation.box(extents=[26, 30, 4])
    roof.apply_translation([10, 0, 9.5])
    mesh = trimesh.util.concatenate([left, right, roof])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    assert _run(mesh) > 100


def test_thin_islands_are_handled_per_part_not_all_or_nothing():
    """조각마다 따로 판단해야 한다. 넓은 조각까지 같이 죽으면 안 된다.

    여백이 필요 없으면(모델 충돌 위험이 없으면) 얇은 조각도 채운다.
    여백이 필요하면 얇은 조각은 건너뛰고 더 작은 구슬에 맡긴다 —
    원본 영역에 큰 구슬을 욱여넣으면 모델을 파고들기 때문이다.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    from pellet_support import SupportBeadParams, support_bead_generate_centers

    params = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08,
                               edge_margin_ratio=0.5)
    wide = shapely_box(0, 0, 10, 10)
    thin = shapely_box(20, 0, 20.4, 10)   # 폭 0.4mm < 여백 0.5mm
    region = unary_union([wide, thin])

    # 여백이 필요한 경우: 넓은 조각은 살고 얇은 조각만 건너뛴다
    guarded = support_bead_generate_centers(region, params, 0, (0.0, 0.0))
    assert [c for c in guarded if c.x < 19.5], "넓은 조각까지 사라졌다"
    assert not [c for c in guarded if c.x >= 19.5]

    # 여백이 필요 없는 경우: 얇은 조각도 채운다
    free = support_bead_generate_centers(region, params, 0, (0.0, 0.0),
                                         edge_margin_override=0.0)
    assert [c for c in free if c.x >= 19.5], "얇은 조각이 통째로 사라졌다"


def test_every_cluster_actually_holds_something_up():
    """'모델 근처'가 아니라 '실제로 받치고 있는가'로 판정해야 한다.

    거리만 보면 선체 옆에 붙어 있기만 한 뭉치도 통과한다. 배 모델에서
    7/7/5/5개짜리 뭉치가 아무것도 안 받치면서 남아 있었다.
    """
    import numpy as np

    from pellet_support.repair import _bead_points, _components

    leg = trimesh.creation.box(extents=[12, 12, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[34, 34, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    ncomp, labels = _components(
        pts, max(contact.pitch_mm(), contact.vertical_neighbor_distance_mm()))
    max_rise = (gen.contact_z_gap_layers * gen.layer_height_mm
                + gen.contact_z_gap_mm
                + contact.bead_diameter_mm * 1.5)

    for comp in range(ncomp):
        cluster = pts[labels == comp]
        origins = cluster.copy()
        origins[:, 2] += 1e-3
        locs, idx, _ = mesh.ray.intersects_location(
            origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)),
            multiple_hits=False)
        holds = False
        if len(idx):
            rise = locs[:, 2] - origins[idx][:, 2]
            holds = bool((rise <= max_rise).any())
        assert holds, f"{len(cluster)}개짜리 뭉치가 아무것도 안 받치고 있다"


def test_orphan_clusters_that_support_nothing_are_removed():
    """베드에 얹혀 있어도 모델을 전혀 안 받치는 뭉치는 지운다.

    배 모델에서 이런 뭉치가 약 50개(구슬 182개) 나왔다. 재료만 쓰고,
    인쇄 중 노즐에 걸려 떨어져 나가면 다른 곳까지 망친다.
    """
    import numpy as np

    from pellet_support.repair import _bead_points, _components

    # 상판 아래 정상 서포터 + 멀리 떨어진 곳의 가짜 뭉치
    leg = trimesh.creation.box(extents=[10, 10, 12])
    leg.apply_translation([0, 0, 6])
    top = trimesh.creation.box(extents=[30, 30, 4])
    top.apply_translation([0, 0, 14])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    query = trimesh.proximity.ProximityQuery(mesh)
    reach = (gen.contact_z_gap_layers * gen.layer_height_mm
             + contact.bead_diameter_mm + gen.xy_clearance_mm)

    ncomp, labels = _components(
        pts, max(contact.pitch_mm(), contact.vertical_neighbor_distance_mm()))
    rng = np.random.default_rng(0)
    for comp in range(ncomp):
        cluster = pts[labels == comp]
        if len(cluster) > 100:
            cluster = cluster[rng.choice(len(cluster), 100, replace=False)]
        nearest = float(np.abs(query.signed_distance(cluster)).min())
        assert nearest <= reach, (
            f"모델에서 {nearest:.1f}mm 떨어진 뭉치가 남아 있다(기준 {reach:.1f}mm)")


def test_orphan_pruning_can_be_disabled():
    """옵션을 끄면 외톨이 뭉치가 남아야 한다.

    밀폐 상자는 이제 서포터가 0개라 이 검사에 못 쓴다(공동이 아예 안 채워짐).
    서포터가 실제로 생기는 모델로 확인한다.
    """
    leg = trimesh.creation.box(extents=[12, 12, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[34, 34, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm(),
                           prune_orphan_clusters=False)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert result.plan is not None
    assert sum(len(l["beads"]) for l in result.plan.layers) > 100


def _symmetric_bridge():
    """좌우로 완벽히 대칭인 모델(다리 2개 + 상판)."""
    parts = []
    for sx in (-1, 1):
        leg = trimesh.creation.box(extents=[6, 20, 14])
        leg.apply_translation([sx * 12, 0, 7])
        parts.append(leg)
    top = trimesh.creation.box(extents=[37, 20, 4])
    top.apply_translation([0, 0, 16])
    parts.append(top)
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def test_support_is_left_right_symmetric_for_symmetric_model():
    """좌우 대칭 모델에는 좌우 대칭인 서포터가 나와야 한다.

    예전에는 격자 원점이 bbox 좌측 끝에 박혀 있어서, 모델 폭이 pitch 의
    정수배가 아니면 좌우가 어긋났다(배 모델 거울상 일치율 2%).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    mesh = _symmetric_bridge()
    cx = 0.5 * float(mesh.bounds[0][0] + mesh.bounds[1][0])
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    dist, _ = cKDTree(pts).query(mirrored)
    matched = (dist < contact.pitch_mm() * 0.25).mean()
    assert matched > 0.95, f"거울상 일치율이 {matched*100:.1f}% 뿐이다"


def test_no_isolated_or_weakly_connected_beads():
    """이웃이 없거나 1~2개뿐인 구슬은 지지 역할을 못 하고 노즐에 걸린다."""
    import numpy as np
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    mesh = _symmetric_bridge()
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    radius = max(contact.pitch_mm(),
                 contact.vertical_neighbor_distance_mm()) * 1.02
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    counts = np.bincount(pairs.ravel(), minlength=len(pts))

    assert (counts == 0).sum() == 0, "완전히 고립된 구슬이 있다"
    weak_ratio = ((counts >= 1) & (counts <= 2)).mean()
    assert weak_ratio < 0.10, f"이웃 1~2개인 구슬이 {weak_ratio*100:.1f}%나 된다"


def test_no_duplicate_beads_at_the_same_spot():
    """겹쳐 놓인 구슬은 그 지점만 과압출되어 노즐이 긁고 지나간다."""
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    mesh = _symmetric_bridge()
    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    too_close = cKDTree(pts).query_pairs(contact.pitch_mm() * 0.5)
    assert len(too_close) == 0, f"겹친 구슬 쌍 {len(too_close)}개"


def _collides_with_model(result, gen, contact, mesh):
    """구슬이 걸치는 z 범위 전체에서 모델과 겹치는 구슬 수를 센다."""
    from shapely.geometry import Point

    from pellet_support.slicing import clean

    det = result.slices
    det_h = float(mesh.extents[2]) / len(det)
    r0 = 0.5 * contact.bead_diameter_mm
    cache = {}

    def sl(k):
        if k not in cache:
            cache[k] = clean(det[k])
        return cache[k]

    hits = 0
    for layer in result.plan.layers:
        zc = layer["z_bottom"] + 0.5 * gen.layer_height_mm
        ks = sorted({int(max(0, min(len(det) - 1, z / det_h)))
                     for z in (zc - r0, zc, zc + r0)})
        for bead in layer["beads"]:
            foot = Point(bead["x"], bead["y"]).buffer(0.5 * bead["d"], 8)
            if any(not sl(k).is_empty and foot.intersects(sl(k)) for k in ks):
                hits += 1
    return hits


def _stepped_model():
    """층마다 단면이 달라지는 모델. 구슬이 z 범위를 걸치며 충돌하기 쉽다."""
    parts = []
    for i, (w, z) in enumerate([(30, 0), (20, 6), (34, 12)]):
        blk = trimesh.creation.box(extents=[w, w, 6])
        blk.apply_translation([0, 0, z + 3])
        parts.append(blk)
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


@pytest.mark.parametrize("nozzle", [1.0, 5.0])
def test_beads_never_collide_with_the_model(nozzle):
    """구슬은 어떤 노즐 크기에서도 모델을 파고들면 안 된다.

    예전에는 세 가지 이유로 파고들었다.
      - 구슬이 걸치는 z 범위를 무시하고 중심 층만 검사
      - 이어 붙인 구슬이 반지름을 무시하고 중심점만 검사(40.9% 충돌)
      - 얇은 조각 폴백이 여백을 무시하고 원본 영역을 그대로 사용(4.9% 충돌)
    """
    mesh = _stepped_model()
    contact, body = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert _collides_with_model(result, gen, contact, mesh) == 0


def test_support_is_mirror_symmetric_for_a_symmetric_model():
    """대칭 모델에는 대칭 서포터가 나와야 한다.

    격자 원점이 bbox 최소 꼭짓점에 있으면 좌우 위상이 어긋나 비대칭이 된다.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    leg = trimesh.creation.box(extents=[10, 10, 12])
    leg.apply_translation([0, 0, 6])
    top = trimesh.creation.box(extents=[40, 20, 4])
    top.apply_translation([0, 0, 14])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    cx = 0.5 * float(mesh.bounds[0][0] + mesh.bounds[1][0])
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    dist, _ = cKDTree(pts).query(mirrored)
    matched = (dist < contact.pitch_mm() * 0.15).sum() / len(pts)
    assert matched > 0.97, f"거울상 일치율 {matched*100:.1f}%"


def test_edge_margin_is_derived_from_collision_not_a_fixed_ratio():
    """여백은 지름의 고정 비율이 아니라 충돌 조건에서 나와야 한다.

    영역은 이미 모델에서 xy_clearance 만큼 떨어져 있으므로, 그보다 더 깎으면
    구슬이 들어갈 자리만 없앤다.
    """
    from shapely.geometry import box as shapely_box

    from pellet_support import SupportBeadParams, support_bead_generate_centers

    params = SupportBeadParams(bead_diameter_mm=1.0, lattice_overlap_ratio=0.08,
                               edge_margin_ratio=0.5)
    strip = shapely_box(0, 0, 20, 1.2)   # 폭 1.2mm

    wide_margin = support_bead_generate_centers(strip, params, 0, (0.0, 0.0))
    tight = support_bead_generate_centers(strip, params, 0, (0.0, 0.0),
                                          edge_margin_override=0.0)
    assert len(tight) > len(wide_margin)


def _bead_hits_model(result, gen, contact, mesh):
    """구슬이 걸치는 z 범위 전체에서 모델과 겹치는 구슬 수를 센다."""
    from shapely.geometry import Point

    from pellet_support.slicing import clean

    det = result.slices
    det_h = float(mesh.extents[2]) / len(det)
    r0 = 0.5 * contact.bead_diameter_mm
    cache = {}

    def sl(k):
        if k not in cache:
            cache[k] = clean(det[k])
        return cache[k]

    hits = 0
    for layer in result.plan.layers:
        zc = layer["z_bottom"] + 0.5 * gen.layer_height_mm
        ks = sorted({int(max(0, min(len(det) - 1, z / det_h)))
                     for z in (zc - r0, zc, zc + r0)})
        for bead in layer["beads"]:
            foot = Point(bead["x"], bead["y"]).buffer(0.5 * bead["d"], 8)
            if any((not sl(k).is_empty) and foot.intersects(sl(k)) for k in ks):
                hits += 1
    return hits


@pytest.mark.parametrize("nozzle", [1.0, 5.0])
def test_beads_never_penetrate_the_model(nozzle):
    """구슬은 위아래로 반지름만큼 뻗는다. 중심 층만 검사하면 다른 층의
    모델을 파고든다. 노즐 5mm 에서 실측 37개(5%)가 충돌했었다."""
    leg = trimesh.creation.box(extents=[12, 12, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[34, 34, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=nozzle)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert result.plan is not None
    assert _bead_hits_model(result, gen, contact, mesh) == 0


def test_support_is_mirror_symmetric_for_a_symmetric_model():
    """대칭 모델이면 서포터도 대칭이어야 한다.

    격자 원점을 bbox 최소 꼭짓점에 두면 좌우 위상이 어긋나 비대칭이 된다.
    중심에 두어야 거울상이 일치한다.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    leg = trimesh.creation.cylinder(radius=6, height=14)
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[34, 34, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    cx = 0.5 * float(mesh.bounds[0][0] + mesh.bounds[1][0])
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    dist, _ = cKDTree(pts).query(mirrored)
    matched = (dist < contact.pitch_mm() * 0.15).sum() / len(pts)
    assert matched > 0.95, f"거울상 일치율 {matched*100:.1f}%"


@pytest.mark.parametrize("nozzle,bead", [(1.0, None), (5.0, 1.75)])
def test_no_bead_is_printed_in_mid_air(nozzle, bead):
    """모든 구슬은 베드·모델·아래층 구슬 중 하나가 받쳐야 한다.

    두 가지 버그가 있었다.
    1. 모델 받침 판정이 '아래 어딘가에 모델이 있는가'(intersects_any)라서,
       선체 한참 위에 뜬 구슬도 저 아래 선체가 걸려 통과했다.
    2. 허공 검사를 맨 처음 한 번만 해서, 이후 정리 단계가 구슬을 지우면
       그 위에 얹혀 있던 구슬이 새로 허공에 떠도 잡지 못했다.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from pellet_support.repair import _bead_points

    leg = trimesh.creation.box(extents=[12, 12, 16])
    leg.apply_translation([0, 0, 8])
    top = trimesh.creation.box(extents=[36, 36, 4])
    top.apply_translation([0, 0, 18])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=nozzle, bead_diameter_mm=bead)
    gen = SupportGenParams(nozzle_diameter_mm=nozzle,
                           layer_height_mm=contact.layer_height_mm())
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert result.plan is not None

    pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
    r0 = 0.5 * contact.bead_diameter_mm
    radius = max(contact.pitch_mm(),
                 contact.vertical_neighbor_distance_mm()) * 1.02
    max_drop = ((gen.contact_z_gap_layers + 1) * gen.layer_height_mm + r0)

    tree = cKDTree(pts)
    below = np.array([
        any(pts[j][2] < p[2] - 1e-6
            for j in tree.query_ball_point(p, radius) if j != i)
        for i, p in enumerate(pts)
    ])
    origins = pts.copy()
    origins[:, 2] -= 1e-3
    locs, idx, _ = mesh.ray.intersects_location(
        origins, np.tile([0, 0, -1.0], (len(pts), 1)), multiple_hits=False)
    on_model = np.zeros(len(pts), dtype=bool)
    if len(idx):
        drop = origins[idx][:, 2] - locs[:, 2]
        on_model[idx[drop <= max_drop]] = True
    on_bed = pts[:, 2] <= r0 * 1.5

    floating = ~(below | on_model | on_bed)
    assert floating.sum() == 0, f"허공에 뜬 구슬 {floating.sum()}개"


def test_gap_filling_adds_smaller_beads_and_shrinks_the_void():
    """큰 구슬이 놓인 영역에도 남는 틈은 더 작은 구슬로 메워야 한다.

    예전에는 '구슬이 하나도 안 들어간 조각'만 다시 시도해서, 경사면 아래처럼
    영역이 좁아지는 곳에 큰 틈이 그대로 남았다.

    측정은 '모델 아랫면에서 가장 가까운 구슬 표면까지의 거리'로 한다.
    기둥 꼭대기의 틈으로 재면, 빈 곳에 구슬이 새로 생기면서 새 꼭대기가
    만들어져 중앙값이 오히려 올라가는 착시가 생긴다.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    leg = trimesh.creation.cylinder(radius=8, height=14)
    leg.apply_translation([0, 0, 7])
    dome = trimesh.creation.icosphere(subdivisions=3, radius=14)
    dome.apply_translation([0, 0, 20])
    mesh = trimesh.util.concatenate([leg, dome])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # 아래를 향한 면(오버행)의 중심점들
    downward = mesh.face_normals[:, 2] < -0.3
    targets = mesh.triangles_center[downward]

    def run(generations):
        contact, body = make_params(nozzle_diameter_mm=5.0,
                                    bead_diameter_mm=1.75)
        gen = SupportGenParams(nozzle_diameter_mm=5.0,
                               layer_height_mm=contact.layer_height_mm(),
                               fill_generations=generations,
                               min_bead_to_nozzle_ratio=0.2)
        result = generate_support(mesh, gen, contact, body,
                                  detail=0, verbose=False)
        pts, diam = [], []
        for layer in result.plan.layers:
            zc = layer["z_bottom"] + 0.5 * gen.layer_height_mm
            for bead in layer["beads"]:
                pts.append((bead["x"], bead["y"], zc))
                diam.append(bead["d"])
        pts = np.asarray(pts)
        diam = np.asarray(diam)
        dist, idx = cKDTree(pts).query(targets)
        n_fine = sum(1 for l in result.plan.layers
                     for b in l["beads"] if b.get("refined"))
        return float(np.median(dist - 0.5 * diam[idx])), n_fine

    gap_off, fine_off = run(0)
    gap_on, fine_on = run(3)

    assert fine_off == 0
    assert fine_on > 0, "세분 구슬이 하나도 안 생겼다"
    assert gap_on < gap_off, f"틈이 줄지 않았다 ({gap_off:.2f} -> {gap_on:.2f})"


def test_no_support_for_a_model_that_needs_none():
    """오버행이 없는 모델에는 구슬이 하나도 안 나와야 한다.

    닫힌 공동의 오버행이 acc 에 남아 계속 아래로 전파되고 flare 로 넓어져,
    속 빈 상자(오버행이 전혀 없어야 정상)에 구슬 25,206개가 모델 바깥에
    생겼었다.
    """
    assert _run(trimesh.creation.box(extents=[20, 20, 20])) == 0
    assert _run(_sealed_hollow_box()) == 0


def test_open_topped_cavity_is_not_treated_as_sealed():
    """위가 트인 통(컵) 안쪽은 구슬을 꺼낼 수 있으므로 막힌 공동이 아니다.

    층별 '구멍' 판정만 쓰면 단면이 고리라는 이유로 막아 버린다. 배 모델에서
    정상 서포터가 675 -> 291개로 줄고 바깥 오버행까지 거리가 6.17 -> 8.55mm
    로 나빠진 원인이었다.
    """
    from pellet_support.regions import _inaccessible_cavities
    from pellet_support.slicing import slice_model

    # 겹쳐 얹기가 아니라 진짜 차집합이어야 '위가 트인 통'이 된다.
    # concatenate 로 만들면 안쪽 실린더가 위쪽에서 solid 로 잡혀 뚜껑이 된다.
    outer = trimesh.creation.cylinder(radius=12, height=20)
    inner = trimesh.creation.cylinder(radius=9, height=22)
    inner.apply_translation([0, 0, 4])
    cup = outer.difference(inner)
    cup.apply_translation([0, 0, -cup.bounds[0][2]])

    slices, _ = slice_model(cup, 0.4, 4000)
    sealed = _inaccessible_cavities(slices, opening_mm=1.0)
    # 위가 트여 있으므로 '꺼낼 수 없는 공동'으로 잡히면 안 된다
    assert sum(s.area for s in sealed) < 1.0


def test_sealed_cavity_is_still_detected():
    """사방이 막힌 공동은 여전히 막힌 것으로 잡혀야 한다."""
    from pellet_support.regions import _inaccessible_cavities
    from pellet_support.slicing import slice_model

    box_mesh = _sealed_hollow_box()
    slices, _ = slice_model(box_mesh, 0.4, 4000)
    sealed = _inaccessible_cavities(slices, opening_mm=1.0)
    assert sum(s.area for s in sealed) > 100.0


def test_fallback_walls_fill_what_beads_cannot():
    """구슬이 못 들어가는 좁은 자리는 일반 서포터(얇은 벽)로 메운다."""
    import numpy as np

    leg = trimesh.creation.cylinder(radius=8, height=14)
    leg.apply_translation([0, 0, 7])
    dome = trimesh.creation.icosphere(subdivisions=3, radius=14)
    dome.apply_translation([0, 0, 20])
    mesh = trimesh.util.concatenate([leg, dome])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    def run(fallback):
        contact, body = make_params(nozzle_diameter_mm=5.0,
                                    bead_diameter_mm=1.75)
        gen = SupportGenParams(nozzle_diameter_mm=5.0,
                               layer_height_mm=contact.layer_height_mm(),
                               min_bead_to_nozzle_ratio=0.2,
                               fallback_solid=fallback)
        result = generate_support(mesh, gen, contact, body,
                                  detail=0, verbose=False)
        walls = [p for l in result.plan.layers
                 for p in l.get("solid_extra", [])]
        return result, walls, contact

    _, walls_off, _ = run(False)
    result_on, walls_on, contact = run(True)

    assert walls_off == []
    assert walls_on, "구슬로 못 채운 자리가 있는데 벽이 하나도 안 생겼다"

    # 벽은 '구슬이 못 들어갈 만큼 좁은' 자리여야 한다
    widths = [4 * p.area / p.length for p in walls_on if p.length > 0]
    assert np.median(widths) < contact.bead_diameter_mm

    # 곡면을 따라 잘린 조각은 꼭짓점이 수천 개라 단순화가 필수다.
    # 안 하면 삼각형이 200배 넘게 폭증한다.
    verts = [len(p.exterior.coords) for p in walls_on]
    assert np.median(verts) < 100, f"단순화가 안 됐다(중앙값 {np.median(verts)})"


def test_fallback_walls_can_be_disabled():
    leg = trimesh.creation.box(extents=[12, 12, 14])
    leg.apply_translation([0, 0, 7])
    top = trimesh.creation.box(extents=[34, 34, 4])
    top.apply_translation([0, 0, 16])
    mesh = trimesh.util.concatenate([leg, top])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    contact, body = make_params(nozzle_diameter_mm=1.0)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm(),
                           fallback_solid=False)
    result = generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert all(not l.get("solid_extra") for l in result.plan.layers)


def test_filler_beads_reach_layers_where_base_bead_cannot_fit():
    """기본 구슬 영역이 완전히 비어도, 세분 구슬이 들어갈 자리가 있으면
    그 층까지 커버리지가 이어져야 한다.

    무한 큐브 꼭대기에서 실측: 같은 층인데 기본구슬(반지름 1.25mm) 기준
    영역은 0mm^2, 세분구슬(반지름 0.5mm) 기준 영역은 772.6mm^2 였다.
    '기본 영역이 비면 그 층은 통째로 건너뛴다'는 조기 종료가 세 군데
    (파이프라인의 collision 클리핑, meshing 의 region-empty continue,
    plan_beads 최상단의 sup.is_empty continue) 있었고, 셋 다 고쳐야
    실제로 커버리지가 늘었다.
    """
    from pellet_support.repair import _bead_points

    outer = trimesh.creation.box(extents=[20, 20, 30])
    cap = trimesh.creation.box(extents=[40, 40, 4])
    cap.apply_translation([0, 0, 17])
    mesh = trimesh.util.concatenate([outer, cap])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    def top_coverage(fill_generations):
        contact, body = make_params(nozzle_diameter_mm=5.0)
        gen = SupportGenParams(nozzle_diameter_mm=5.0,
                               layer_height_mm=contact.layer_height_mm(),
                               fill_generations=fill_generations)
        result = generate_support(mesh, gen, contact, body,
                                  detail=0, verbose=False)
        pts, _ = _bead_points(result.plan, gen.layer_height_mm, 0.0)
        return float(pts[:, 2].max()) if len(pts) else 0.0

    top_off = top_coverage(0)     # 세분화 완전히 끔(비교 기준)
    top_on = top_coverage(3)      # 기본값
    assert top_on >= top_off, (
        f"세분화를 켰는데 커버리지가 늘지 않았다 ({top_off:.1f} -> {top_on:.1f})")


def test_fast_precheck_blocks_oversized_requests_in_seconds():
    """정밀 계산(슬라이싱+영역 계산)을 다 돌리기 전에, 확실히 과한 경우는
    삼각형 법선만으로 몇 초 안에 미리 막아야 한다.

    나뭇가지 모양처럼 복잡한 모델은 정밀 계산 자체에 68초가 걸렸고,
    그동안 브라우저/프록시가 연결을 끊어 '서버가 죽었다'로 보였다.
    """
    import time

    from pellet_support.validation import TooManyBeadsError

    # 표면적이 넓고 오버행이 많은 구체 -> 아주 작은 구슬이면 확실히 과함
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=40)
    mesh.apply_translation([0, 0, 40])

    contact, body = make_params(nozzle_diameter_mm=1.0, bead_diameter_mm=0.3)
    gen = SupportGenParams(nozzle_diameter_mm=1.0,
                           layer_height_mm=contact.layer_height_mm(),
                           max_beads=1000)
    t0 = time.time()
    with pytest.raises(TooManyBeadsError):
        generate_support(mesh, gen, contact, body, detail=0, verbose=False)
    assert time.time() - t0 < 10, "사전 점검이 10초 안에 끝나야 한다"
