#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""서포터가 필요한 테스트 모델 몇 개를 만든다.

    python examples/make_test_models.py
    pellet-support examples/out/table.stl --nozzle 1.0 --preview -1
"""

from __future__ import annotations

import os

import trimesh

OUT = os.path.join(os.path.dirname(__file__), "out")


def table() -> trimesh.Trimesh:
    """다리 4개 위에 상판. 상판 아랫면 전체가 오버행."""
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = trimesh.creation.box(extents=[5, 5, 12])
            leg.apply_translation([sx * 10, sy * 6, 6])
            parts.append(leg)
    top = trimesh.creation.box(extents=[30, 20, 4])
    top.apply_translation([0, 0, 14])
    parts.append(top)
    return trimesh.util.concatenate(parts)


def lollipop() -> trimesh.Trimesh:
    """막대 위의 구. 곡면 아랫면이 점점 넓어지는 오버행."""
    ball = trimesh.creation.icosphere(subdivisions=3, radius=12)
    ball.apply_translation([0, 0, 26])
    stick = trimesh.creation.cylinder(radius=3, height=16)
    stick.apply_translation([0, 0, 8])
    return trimesh.util.concatenate([ball, stick])


def overhang_ramp() -> trimesh.Trimesh:
    """여러 각도의 경사면. overhang-angle 튜닝 확인용."""
    parts = []
    for i, angle in enumerate((20, 35, 50, 65)):
        wall = trimesh.creation.box(extents=[6, 20, 20])
        wall.apply_transform(
            trimesh.transformations.rotation_matrix(
                __import__("math").radians(angle), [0, 1, 0]
            )
        )
        wall.apply_translation([i * 14 - 21, 0, 12])
        parts.append(wall)
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("table", table), ("lollipop", lollipop),
                     ("ramp", overhang_ramp)):
        mesh = fn()
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
        path = os.path.join(OUT, f"{name}.stl")
        mesh.export(path)
        print(f"{path}  ({len(mesh.faces)} faces, {mesh.extents.round(1)} mm)")


if __name__ == "__main__":
    main()
