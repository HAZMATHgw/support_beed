"""Conservative sphere and branch collisions against a stack of model slices.

Each slice occupies the slab between neighbouring height midpoints. The outer
slabs extend by half their nearest spacing; a lone slice represents a plane,
because its thickness cannot be inferred. This is a sliced-model approximation,
not a substitute for checking the original mesh at a finer slicing resolution.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


class SliceCollision:
    """Reuse slab bounds and point distances while growing tree branches.

    ``xy_clearance`` applies horizontally to each actual sphere cross-section.
    It does not extend a sphere vertically into a model above its top. Exact
    tangency is allowed; overlapping interiors or insufficient clearance are not.
    """

    def __init__(self, model_slices, heights, xy_clearance):
        self.heights = tuple(float(z) for z in heights)
        self.xy_clearance = self._nonnegative(xy_clearance, "xy_clearance")
        source = list(model_slices)
        if len(source) != len(self.heights):
            raise ValueError("model_slices and heights must have the same length")
        if any(not math.isfinite(z) for z in self.heights):
            raise ValueError("heights must be finite")
        if any(a >= b for a, b in zip(self.heights, self.heights[1:])):
            raise ValueError("heights must be strictly increasing")
        self.slices = tuple(
            Polygon() if geom is None or geom.is_empty else
            geom if geom.is_valid else geom.buffer(0)
            for geom in source
        )
        if len(self.heights) > 1:
            mids = [(a + b) * 0.5 for a, b in zip(self.heights, self.heights[1:])]
            first = self.heights[0] - (self.heights[1] - self.heights[0]) * 0.5
            last = self.heights[-1] + (self.heights[-1] - self.heights[-2]) * 0.5
            self.slab_bottoms = (first, *mids)
            self.slab_tops = (*mids, last)
        else:
            self.slab_bottoms = self.slab_tops = self.heights
        self._bounds = tuple(None if geom.is_empty else geom.bounds for geom in self.slices)
        self._distances = OrderedDict()
        self._distance_cache_limit = 16384

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    @staticmethod
    def _xyz(values):
        values = tuple(float(value) for value in values)
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("coordinates must contain three finite numbers")
        return values

    def _indices(self, lower, upper):
        return range(bisect_left(self.slab_tops, lower),
                     bisect_right(self.slab_bottoms, upper))

    def _point_distance(self, index, x, y):
        key = (index, x, y)
        cached = self._distances.get(key)
        if cached is not None:
            self._distances.move_to_end(key)
            return cached
        point = Point(x, y)
        model = self.slices[index]
        distance = model.distance(point)
        result = (distance, distance == 0.0 and model.contains(point))
        self._distances[key] = result
        if len(self._distances) > self._distance_cache_limit:
            self._distances.popitem(last=False)
        return result

    def sphere_clear(self, x, y, z, radius):
        """Whether a sphere respects the model and horizontal clearance."""
        x, y, z = self._xyz((x, y, z))
        radius = self._nonnegative(radius, "radius")
        for index in self._indices(z - radius, z + radius):
            bounds = self._bounds[index]
            if bounds is None:
                continue
            dz = max(self.slab_bottoms[index] - z, z - self.slab_tops[index], 0.0)
            # A positive-radius sphere merely tangent to a slab has no interior
            # in that slab. In particular it needs no extra horizontal gap there.
            if radius > 0 and dz >= radius:
                continue
            section_radius = math.sqrt(max(0.0, radius * radius - dz * dz))
            required = section_radius + self.xy_clearance
            if (x + required < bounds[0] or x - required > bounds[2]
                    or y + required < bounds[1] or y - required > bounds[3]):
                continue
            distance, inside = self._point_distance(index, x, y)
            if inside or distance < required:
                return False
        return True

    def free_region(self, region, z, radius):
        """Return the polygon where sphere centres at ``z`` can be sampled.

        Remove each intersected slab's model offset by the sphere's actual
        cross-section radius plus XY clearance. A tangent slab is skipped just
        as in :meth:`sphere_clear`. Circular buffers use a tiny conservative
        expansion so their polygon chords do not leave invalid corner samples.
        """
        _, _, z = self._xyz((0.0, 0.0, z))
        radius = self._nonnegative(radius, "radius")
        if region is None or region.is_empty:
            return Polygon()
        if not region.is_valid or region.geom_type not in ("Polygon", "MultiPolygon"):
            region = region.buffer(0)
        if region.is_empty:
            return Polygon()
        region_bounds = region.bounds
        forbidden = []
        quad_segs = 32
        # Shapely's round buffers are inscribed polygons. Expanding by the
        # secant of half an arc segment makes them enclose the exact offset.
        buffer_scale = 1.0 / math.cos(math.pi / (4.0 * quad_segs))
        for index in self._indices(z - radius, z + radius):
            bounds = self._bounds[index]
            if bounds is None:
                continue
            dz = max(self.slab_bottoms[index] - z, z - self.slab_tops[index], 0.0)
            if radius > 0 and dz >= radius:
                continue
            section_radius = math.sqrt(max(0.0, radius * radius - dz * dz))
            required = (section_radius + self.xy_clearance) * buffer_scale
            if (region_bounds[2] < bounds[0] - required
                    or region_bounds[0] > bounds[2] + required
                    or region_bounds[3] < bounds[1] - required
                    or region_bounds[1] > bounds[3] + required):
                continue
            model = self.slices[index]
            forbidden.append(model.buffer(required, quad_segs=quad_segs)
                             if required else model)
        if not forbidden:
            return region
        result = region.difference(unary_union(forbidden))
        if not result.is_valid or result.geom_type not in ("Polygon", "MultiPolygon"):
            result = result.buffer(0)
        return Polygon() if result.is_empty else result

    @staticmethod
    def _clip_leq(lo, hi, intercept, slope, limit):
        """Intersect a parameter interval with intercept + slope*t <= limit."""
        if slope > 0:
            hi = min(hi, (limit - intercept) / slope)
        elif slope < 0:
            lo = max(lo, (limit - intercept) / slope)
        elif intercept > limit:
            return None
        return (lo, hi) if lo <= hi else None

    @staticmethod
    def _section_radius(z, dz, radius, dr, bottom, top, lo, hi):
        """Maximum sphere cross-section radius in one slab over t in [lo, hi]."""
        cuts = [lo, hi]
        if dz:
            cuts.extend(t for t in ((bottom - z) / dz, (top - z) / dz)
                        if lo < t < hi)
        cuts.sort()
        maximum = 0.0
        for a, b in zip(cuts, cuts[1:]):
            middle_z = z + dz * (a + b) * 0.5
            plane = bottom if middle_z < bottom else top if middle_z > top else None
            # Squared cross-section radius is quadratic outside the slab and
            # r(t)^2 inside it. Check its endpoints and any concave vertex.
            offset = z - plane if plane is not None else 0.0
            slope = dz if plane is not None else 0.0
            quadratic = dr * dr - slope * slope
            linear = 2.0 * (radius * dr - offset * slope)
            candidates = [a, b]
            if quadratic < 0:
                vertex = -linear / (2.0 * quadratic)
                if a < vertex < b:
                    candidates.append(vertex)
            for t in candidates:
                squared = (radius + dr * t) ** 2 - (offset + slope * t) ** 2
                maximum = max(maximum, squared)
        return math.sqrt(maximum)

    def _envelope_clear(self, start, end, start_radius, end_radius):
        """A slab-wise XY capsule contains the entire swept, tapered branch."""
        dx, dy, dz = (b - a for a, b in zip(start, end))
        dr = end_radius - start_radius
        lower = min(start[2] - start_radius, end[2] - end_radius)
        upper = max(start[2] + start_radius, end[2] + end_radius)
        for index in self._indices(lower, upper):
            bounds = self._bounds[index]
            if bounds is None:
                continue
            bottom, top = self.slab_bottoms[index], self.slab_tops[index]
            interval = self._clip_leq(0.0, 1.0, start[2] - start_radius, dz - dr, top)
            if interval is None:
                continue
            interval = self._clip_leq(*interval, -start[2] - start_radius, -dz - dr, -bottom)
            if interval is None:
                continue
            lo, hi = interval
            section = self._section_radius(start[2], dz, start_radius, dr, bottom, top, lo, hi)
            if section == 0 and max(start_radius, end_radius) > 0:
                continue
            required = section + self.xy_clearance
            a = (start[0] + dx * lo, start[1] + dy * lo)
            b = (start[0] + dx * hi, start[1] + dy * hi)
            if (max(a[0], b[0]) + required < bounds[0]
                    or min(a[0], b[0]) - required > bounds[2]
                    or max(a[1], b[1]) + required < bounds[1]
                    or min(a[1], b[1]) - required > bounds[3]):
                continue
            if a == b:
                distance, inside = self._point_distance(index, *a)
                if inside or distance < required:
                    return False
            else:
                path = LineString((a, b))
                model = self.slices[index]
                if model.distance(path) < required or (required == 0 and model.intersects(path)):
                    return False
        return True

    def edge_clear(self, start_xyz, end_xyz, start_radius, end_radius):
        """Check every sphere swept along an edge with linearly varying radius.

        Capsule bounds, rather than isolated samples, prevent thin walls from
        falling between checks. Ambiguous bounds are bisected up to 16 times;
        unresolved near-contact cases are conservatively rejected.
        """
        start, end = self._xyz(start_xyz), self._xyz(end_xyz)
        start_radius = self._nonnegative(start_radius, "start_radius")
        end_radius = self._nonnegative(end_radius, "end_radius")
        if not self.sphere_clear(*start, start_radius) or not self.sphere_clear(*end, end_radius):
            return False
        pending = [(start, end, start_radius, end_radius, 0)]
        while pending:
            a, b, ra, rb, depth = pending.pop()
            if self._envelope_clear(a, b, ra, rb):
                continue
            middle = tuple((x + y) * 0.5 for x, y in zip(a, b))
            rm = (ra + rb) * 0.5
            if depth >= 16 or not self.sphere_clear(*middle, rm):
                return False
            pending.append((middle, b, rm, rb, depth + 1))
            pending.append((a, middle, ra, rm, depth + 1))
        return True

    def max_radius(self, x, y, z, upper):
        """Largest clear radius up to ``upper``, to relative precision 2**-32.

        Return zero when even the point has insufficient horizontal clearance.
        """
        x, y, z = self._xyz((x, y, z))
        upper = self._nonnegative(upper, "upper")
        if not self.sphere_clear(x, y, z, 0.0):
            return 0.0
        if self.sphere_clear(x, y, z, upper):
            return upper
        lower = 0.0
        for _ in range(32):
            middle = (lower + upper) * 0.5
            if self.sphere_clear(x, y, z, middle):
                lower = middle
            else:
                upper = middle
        return lower
