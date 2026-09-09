# -*- coding: utf-8 -*-
"""충전 기하 파라미터.

이 모듈이 저장소 전체의 유일한 '설계 변수' 정의 지점이다. 구 지름 D 와
겹침량 delta 두 개만 정하면 pitch, 층높이, 접촉원 크기, 압출량, 충전율이
전부 여기서 유도된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

EPSILON = 1e-9

#: 모서리 1인 정사면체의 높이 = sqrt(2/3).
#:
#: 서로 맞닿은 세 구가 만드는 오목한 자리(hollow)에 네 번째 구가 앉으면,
#: 그 중심은 아래 세 구의 중심 평면에서 ``pitch * TETRA_HEIGHT`` 만큼 위에
#: 놓인다. 즉 층높이는 선택하는 값이 아니라 pitch 가 정해지는 순간 강제된다.
TETRA_HEIGHT = 0.8164965809277260


class SupportBeadRegion:
    """bead 가 놓이는 위치의 종류."""

    CONTACT = "contact"  # 모델 아랫면과 닿는 껍질
    BODY = "body"        # 서포터 몸통


@dataclass
class SupportBeadParams:
    """C++ ``SupportBeadParams`` 와 1:1 대응."""

    bead_diameter_mm: float = 1.0
    #: delta. 이웃 구에 얼마나 눌려 들어가는지를 지름 대비 비율로.
    #: 0 이면 점접촉(고정력 0), 0.12 를 넘으면 사실상 통짜가 된다.
    lattice_overlap_ratio: float = 0.06
    #: 가로(평면) 방향 겹침. None 이면 lattice_overlap_ratio 를 그대로 쓴다.
    #:
    #: 플레이트가 좌우로 흔들릴 때 무너지느냐는 '평면 넥'이 결정한다. 층
    #: 간격을 pitch*sqrt(2/3) 으로 고정하면 겹침이 모든 방향에 똑같이 걸려서,
    #: 가로를 강하게 하려면 세로까지 같이 강해지고 결국 통짜가 되어 펠릿으로
    #: 부서지지 않는다. 가로/세로를 따로 두면 구슬 크기를 키우지 않고도
    #: 가로만 강하게, 세로는 약하게(=분리는 쉽게) 만들 수 있다.
    lateral_overlap_ratio: Optional[float] = None
    #: 세로(층 사이) 방향 겹침. None 이면 lattice_overlap_ratio.
    vertical_overlap_ratio: Optional[float] = None
    edge_margin_ratio: float = 0.5
    #: 압출 선분 길이 / 지름. 짧을수록 구에 가깝다.
    segment_ratio: float = 0.15
    #: True = 아래층 hollow 에 안착(배위수 12), False = 수직 기둥(배위수 8)
    stagger_layers: bool = True
    #: 2 = ABAB(hcp), 3 = ABCABC(fcc)
    stagger_period: int = 3
    snake_order: bool = True
    alternate_contact_angle: bool = True
    flow_multiplier: float = 1.0

    # -- 유도값 ------------------------------------------------------------

    def lateral_overlap(self) -> float:
        return (self.lateral_overlap_ratio
                if self.lateral_overlap_ratio is not None
                else self.lattice_overlap_ratio)

    def vertical_overlap(self) -> float:
        return (self.vertical_overlap_ratio
                if self.vertical_overlap_ratio is not None
                else self.lattice_overlap_ratio)

    def pitch_mm(self) -> float:
        """같은 층 안에서 구 중심 사이 거리(가로 간격)."""
        return self.bead_diameter_mm * (1.0 - self.lateral_overlap())

    def vertical_neighbor_distance_mm(self) -> float:
        """아래층 세 구슬과의 중심 거리."""
        return self.bead_diameter_mm * (1.0 - self.vertical_overlap())

    def layer_height_mm(self) -> float:
        """이 충전이 성립하기 위해 프로파일이 반드시 써야 하는 층높이.

        가로/세로 겹침이 같으면 기존과 똑같이 pitch*sqrt(2/3) 이 나온다.
        다르면, 아래 세 구슬과의 거리가 vertical_neighbor_distance 가 되도록
        층 간격을 역산한다: h = sqrt(d_v^2 - pitch^2/3).
        """
        pitch = self.pitch_mm()
        if not self.stagger_layers:
            return self.vertical_neighbor_distance_mm()
        d_v = self.vertical_neighbor_distance_mm()
        inner = d_v * d_v - pitch * pitch / 3.0
        if inner <= 0:
            # 가로를 너무 눌러서 세로로 층을 쌓을 수 없는 상태.
            # 등방(원래 기하)으로 되돌린다.
            return pitch * TETRA_HEIGHT
        return math.sqrt(inner)

    def contact_disc_radius_mm(self) -> float:
        """같은 층 이웃과 눌려 생기는 평평한 원판의 반지름(가로 넥)."""
        r, d = 0.5 * self.bead_diameter_mm, self.pitch_mm()
        return math.sqrt(max(0.0, r * r - 0.25 * d * d))

    def vertical_disc_radius_mm(self) -> float:
        """위아래 층 이웃과의 넥 반지름(세로 넥)."""
        r = 0.5 * self.bead_diameter_mm
        d = self.vertical_neighbor_distance_mm()
        return math.sqrt(max(0.0, r * r - 0.25 * d * d))

    def lateral_neck_area_mm2(self) -> float:
        a = self.contact_disc_radius_mm()
        return math.pi * a * a

    def vertical_neck_area_mm2(self) -> float:
        a = self.vertical_disc_radius_mm()
        return math.pi * a * a

    def coordination_number(self) -> int:
        """한 구가 닿는 이웃의 수."""
        return 12 if self.stagger_layers else 8

    def bead_volume_mm3(self) -> float:
        """구 부피에서 이웃이 눌러 없앤 구관(spherical cap)들을 뺀 값."""
        r = 0.5 * self.bead_diameter_mm
        overlap = max(0.0, self.bead_diameter_mm - self.pitch_mm())
        h_cap = 0.5 * overlap
        v_cap = math.pi * h_cap * h_cap * (3.0 * r - h_cap) / 3.0
        return max(
            0.0,
            4.0 / 3.0 * math.pi * r ** 3 - self.coordination_number() * v_cap,
        )

    def mm3_per_mm(self) -> float:
        """압출 선분에 실어야 할 단위길이당 부피.

        선분 길이가 아니라 목표 구 부피에서 역산한다. 그래야 구 크기가
        세그먼트 길이에 끌려다니지 않는다.
        """
        seg = max(EPSILON, self.bead_diameter_mm * self.segment_ratio)
        return self.flow_multiplier * self.bead_volume_mm3() / seg

    def packing_fraction(self) -> float:
        """단위 격자 셀 대비 재료가 차지하는 비율."""
        pitch, height = self.pitch_mm(), self.layer_height_mm()
        cell = math.sqrt(3.0) * 0.5 * pitch * pitch * height
        return self.bead_volume_mm3() / cell if cell > 0 else 0.0


#: 표준 FDM 관행: 층높이는 보통 노즐 지름의 25~75% 를 쓰지, 노즐 지름
#: 그대로 쓰지 않는다(0.4mm 노즐에 0.2mm 층높이가 흔한 이유). 구 지름도
#: 같은 논리로, 굵은 펠릿 노즐이라고 구까지 그만큼 굵을 필요는 없다.
#: 기본값 0.5 는 "노즐의 절반 크기 구슬"이라는 뜻.
DEFAULT_BEAD_TO_NOZZLE_RATIO = 0.5


def support_bead_contact_params(
    nozzle_diameter_mm: float,
    bead_diameter_mm: Optional[float] = None,
) -> SupportBeadParams:
    """모델 아랫면과 닿는 껍질. 조금 더 눌러 붙여 전단에 견디게 한다.

    ``bead_diameter_mm`` 을 비워두면 노즐 지름의
    :data:`DEFAULT_BEAD_TO_NOZZLE_RATIO` 를 쓴다. 노즐이 굵을수록
    구를 노즐과 똑같이 키우면 gap/interface/edge_margin 이 전부 비드
    지름에 비례해서 커져, 굵은 펠릿 노즐에서 서포터 위쪽이 통째로
    비는 사각지대가 생긴다(55.7mm 모델 기준 노즐 8mm 일 때 32%).
    비드를 노즐보다 작게 잡으면 이 사각지대가 그만큼 줄어든다.
    """
    # 0.0 도 명시적으로 지정한 값이므로 None 인지로만 판단한다
    # (예전에는 falsy 라서 조용히 자동값으로 바뀌었다).
    d = bead_diameter_mm if bead_diameter_mm is not None else nozzle_diameter_mm * DEFAULT_BEAD_TO_NOZZLE_RATIO
    return SupportBeadParams(
        bead_diameter_mm=d,
        lattice_overlap_ratio=0.08,
        edge_margin_ratio=0.4,
    )


def support_bead_body_params(
    nozzle_diameter_mm: float,
    bead_ratio: float = 0.97,
    bead_diameter_mm: Optional[float] = None,
) -> SupportBeadParams:
    """몸통. 구를 조금 더 작게 만들어 겹침을 낮추면 펠릿으로 잘 부서진다.

    ``bead_diameter_mm`` 을 비워두면 contact 와 같은 규칙(노즐의
    :data:`DEFAULT_BEAD_TO_NOZZLE_RATIO`)을 쓰고, 거기서 ``bead_ratio``
    만큼 한 번 더 줄인다.
    """
    base = bead_diameter_mm if bead_diameter_mm is not None else nozzle_diameter_mm * DEFAULT_BEAD_TO_NOZZLE_RATIO
    return SupportBeadParams(
        bead_diameter_mm=base * bead_ratio,
        lattice_overlap_ratio=0.04,
        edge_margin_ratio=0.5,
    )


def unify_lattice(
    contact: SupportBeadParams, body: SupportBeadParams
) -> Tuple[SupportBeadParams, SupportBeadParams]:
    """격자는 오브젝트당 하나뿐이라는 제약을 강제한다.

    contact 의 pitch 를 기준으로 삼고, body 는 같은 pitch 위에서 구 지름만
    줄여 겹침을 낮춘다. body 의 delta 를 직접 지정하게 두면 pitch 가 갈라져
    위아래 층이 서로의 hollow 에 안 떨어진다.
    """
    pitch = contact.pitch_mm()
    d_vert = contact.vertical_neighbor_distance_mm()
    body_d = body.bead_diameter_mm or contact.bead_diameter_mm
    overrides = dict(
        lattice_overlap_ratio=1.0 - pitch / body_d,
        stagger_layers=contact.stagger_layers,
        stagger_period=contact.stagger_period,
    )
    # 비등방 충전에서는 가로/세로 겹침이 pitch 보다 우선하므로 이것도 같이
    # 맞춰야 한다. 안 그러면 body 가 자기 지름으로 따로 pitch 를 계산해서
    # 격자가 갈라지고, 위아래 층이 서로의 hollow 에 안 떨어진다.
    if contact.lateral_overlap_ratio is not None or body.lateral_overlap_ratio is not None:
        overrides["lateral_overlap_ratio"] = 1.0 - pitch / body_d
    if contact.vertical_overlap_ratio is not None or body.vertical_overlap_ratio is not None:
        overrides["vertical_overlap_ratio"] = 1.0 - d_vert / body_d
    body = replace(body, **overrides)
    return contact, body


@dataclass
class SupportGenParams:
    """서포터 '영역'을 찾는 단계의 파라미터."""

    nozzle_diameter_mm: float = 1.0
    layer_height_mm: float = 0.77  # bead 기하에서 자동 계산됨
    #: 오버행 탐지용 슬라이싱 두께. None 이면 자동(구슬 층높이와 0.4mm 중 작은 값).
    #: 구슬 격자 간격과 분리해 두어야 굵은 펠릿에서도 오버행을 놓치지 않는다.
    detection_layer_height_mm: float = None
    max_detection_layers: int = 3000
    overhang_angle_deg: float = 45.0
    contact_z_gap_layers: int = 1
    xy_clearance_mm: float = 0.8
    contact_layers: int = 2
    #: 베드에 닿는 첫 층을 구슬 대신 통판으로 채울 층 수. 통판은 펠릿으로
    #: 부서지지 않고 큰 판때기로 남으므로 기본은 0(전부 구슬)이다.
    solid_first_layers: int = 0
    min_island_area_mm2: float = 2.0
    support_on_build_plate_only: bool = False
    #: 모델 내부의 닫힌 공동까지 서포터로 채울지. 기본은 채우지 않는다.
    #: 속이 빈 상자나 선체 안쪽처럼 사방이 막힌 공간에 구슬을 채우면 출력 후
    #: 꺼낼 방법이 없어서 재료와 시간만 버리고 무게만 늘어난다.
    allow_internal_supports: bool = False
    #: 한 탐지 층 내려갈 때마다 서포터를 바깥으로 넓히는 양(mm).
    #: 바닥을 넓혀 좌우 흔들림에 버티게 하고, 따로 시작한 기둥들이 내려오며
    #: 하나로 합쳐지게 한다. 0 이면 예전처럼 수직 기둥만 만든다.
    #: 공중에 뜬 구슬 덩어리를 아래로 이어 붙일지. 지우는 것보다 낫다.
    #: 구슬이 하나도 안 들어간 좁은 조각을 더 작은 구슬로 다시 채울지.
    refine_thin_regions: bool = True
    #: 인쇄 가능한 최소 구슬 지름(mm). None 이면 노즐 * 아래 비율.
    min_bead_diameter_mm: Optional[float] = None
    min_bead_to_nozzle_ratio: float = 0.35
    #: 모델을 전혀 받치지 않는 외톨이 구슬 뭉치를 지울지.
    prune_orphan_clusters: bool = True
    stitch_floating: bool = True
    #: 이어 붙일 때 기둥을 몇 칸 간격으로 내릴지(1=전부, 클수록 성김).
    #: 1 이면 구슬이 3% 늘어나는 대신 이어 붙인 구간이 훨씬 촘촘해진다
    #: (이웃 3개 이하인 '매달린' 구슬 3.6% -> 3.0%).
    stitch_stride: int = 1
    flare_mm_per_layer: float = 0.08
    #: 아래에 받쳐줄 것이 없는 구슬을 제거할지. 공중에 뜬 구슬은
    #: 인쇄되지 않고 노즐에 끌려다니며 출력을 망친다.
    prune_unsupported: bool = True
    #: 바닥 브림: 아래쪽 몇 mm 구간을 바깥으로 넓혀 접지 면적을 키운다.
    #: 흔들림에 넘어지는지는 바닥 폭이 결정하는데, flare 로 전체를 넓히면
    #: 구슬 수가 폭증하므로 바닥 근처에만 더하는 편이 훨씬 싸다.
    brim_mm: float = 3.0
    brim_height_mm: float = 4.0
    #: 구슬 수 상한. 넘으면 메모리가 터지기 전에 미리 막는다.
    #: 구슬 수 상한. None 이면 구슬 면 수와 메모리 예산에서 자동 계산.
    max_beads: Optional[int] = None
    max_layers: int = 4000
