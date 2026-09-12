"""Output and cache regressions for the patch-22 tree performance changes."""

import math
from pathlib import Path
import sys

import pytest
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pellet_support.skeleton import _bead_layer_offsets, _nudge_out_of_model


@pytest.mark.parametrize("x,y,diameter,expected", [
    (-2, 5, 0.4, (-2, 5, 1, 0.4)),
    (-0.1, 5, 0.4, (-0.4700000000000001, 5.0, 1, 0.4)),
    (1, 5, 0.4, (-0.4700000000000001, 5.0, 1, 0.4)),
    (-0.1, -0.1, 0.4, (-0.3431853069306006, -0.3204103144836018, 1, 0.4)),
    (-0.1, 5, 0.8, (-0.6900000000000001, 5.0, 1, 0.8)),
])
def test_nudge_retains_patch22_wall_and_corner_results(x, y, diameter, expected):
    heights = [0.25, 0.75, 1.25, 1.75]
    slices = [box(0, 0, 10, 10)] * len(heights)
    for cache in (None, {}):
        result = _nudge_out_of_model(x, y, 1, diameter, slices, heights, 0.1, _cache=cache)
        assert result == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("heights,z,diameter,expected_interval", [
    ([0, 2, 4], 2, 2, (0, 1)),  # both endpoints are tied: choose earlier layers
    ([0, 2, 4], -10, 0.4, (0, 0)),
    ([0, 2, 4], 10, 0.4, (2, 2)),
    ([0, 2, 2, 4], 2, 0, (1, 1)),  # repeated equal heights also choose first
    ([0, 2, 2], 10, 0.4, (1, 1)),
])
def test_nearest_slice_lookup_matches_patch22_tie_and_endpoint_rules(
        heights, z, diameter, expected_interval):
    cache = {}
    _nudge_out_of_model(-100, 0, z, diameter, [box(0, 0, 1, 1)] * len(heights),
                        heights, 0.1, _cache=cache)
    intervals = [key for key in cache if len(key) == 2]
    assert intervals == [expected_interval]


def test_repeated_beads_reuse_buffer_and_different_radii_get_separate_buffers(monkeypatch):
    original = BaseGeometry.buffer
    calls = []

    def counted_buffer(self, distance, *args, **kwargs):
        calls.append(distance)
        return original(self, distance, *args, **kwargs)

    monkeypatch.setattr(BaseGeometry, "buffer", counted_buffer)
    cache = {}
    heights = [0.25, 0.75, 1.25, 1.75]
    slices = [box(0, 0, 10, 10)] * len(heights)
    for _ in range(5):
        _nudge_out_of_model(-0.1, 5, 1, 0.4, slices, heights, 0.1, _cache=cache)
    assert len(calls) == 1
    _nudge_out_of_model(-0.1, 5, 1, 0.6, slices, heights, 0.1, _cache=cache)
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_empty_slices_preserve_coordinates_without_building_a_buffer():
    assert _nudge_out_of_model(2, 3, 1, 0.4, [Polygon()] * 2, [0.5, 1.5], 0.1,
                               _cache={}) == (2, 3, 1, 0.4)


def test_cached_hexagonal_offsets_keep_patch22_positions_order_and_shell():
    height = 0.9 * math.sqrt(3) / 2
    expected = [(-0.45, -height), (0.45, -height),
                (-0.9, 0), (0, 0), (0.9, 0),
                (-0.45, height), (0.45, height)]
    assert _bead_layer_offsets(1.0, 0.9, False) == expected
    assert _bead_layer_offsets(1.0, 0.9, False, shell=0.2) == [p for p in expected if p != (0, 0)]
    modified = _bead_layer_offsets(1.0, 0.9, False)
    modified.clear()
    assert _bead_layer_offsets(1.0, 0.9, False) == expected
    assert _bead_layer_offsets(0.1, 0.9, False) == [(0, 0)]
