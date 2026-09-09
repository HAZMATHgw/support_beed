# -*- coding: utf-8 -*-
"""최밀충전 구형 펠릿 서포터 생성기."""

from .lattice import (
    SupportBeadCenter,
    bead_angle,
    support_bead_generate_centers,
)
from .meshing import BeadPlan, plan_beads, plan_to_mesh
from .params import (
    EPSILON,
    TETRA_HEIGHT,
    SupportBeadParams,
    SupportBeadRegion,
    SupportGenParams,
    support_bead_body_params,
    support_bead_contact_params,
    unify_lattice,
)
from .pipeline import SupportResult, generate_support, make_params
from .regions import build_support_regions
from .report import (
    export_plan_json,
    export_preview,
    load_mesh,
    measure_packing,
    report_packing,
    verify_packing,
)
from .slicing import slice_model

__version__ = "0.1.0"

__all__ = [
    "TETRA_HEIGHT",
    "EPSILON",
    "SupportBeadParams",
    "SupportBeadRegion",
    "SupportGenParams",
    "SupportBeadCenter",
    "BeadPlan",
    "SupportResult",
    "support_bead_contact_params",
    "support_bead_body_params",
    "unify_lattice",
    "make_params",
    "slice_model",
    "build_support_regions",
    "support_bead_generate_centers",
    "bead_angle",
    "plan_beads",
    "plan_to_mesh",
    "generate_support",
    "load_mesh",
    "report_packing",
    "verify_packing",
    "measure_packing",
    "export_preview",
    "export_plan_json",
    "__version__",
]
