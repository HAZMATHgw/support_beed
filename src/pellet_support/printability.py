"""Bottom-up support paths for the final bead coordinates.

A connected component is not necessarily printable: a chain can hang down from
an otherwise supported branch. Only lower beads and actual upward model faces
can support a bead. XY clearance is never a permitted air gap below a foot.
"""

from collections import deque
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def _graph(beads):
    xyz, radii = beads[:, :3], beads[:, 3] * 0.5
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(1.96 * radii.max(), output_type="ndarray")
    if len(pairs):
        delta = xyz[pairs[:, 1]] - xyz[pairs[:, 0]]
        touching = np.linalg.norm(delta, axis=1) < 0.98 * radii[pairs].sum(axis=1)
        printable = np.linalg.norm(delta[:, :2], axis=1) <= np.abs(delta[:, 2]) + 1e-9
        rising = np.abs(delta[:, 2]) > 1e-8
        pairs = pairs[touching & printable & rising]
        reverse = xyz[pairs[:, 0], 2] > xyz[pairs[:, 1], 2]
        pairs[reverse] = pairs[reverse, ::-1]
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(beads), len(beads))).tocsr()
    return tree, graph


def _grow_supported(graph, supported, starts):
    queue = deque(int(i) for i in starts if not supported[i])
    for i in queue:
        supported[i] = True
    while queue:
        i = queue.popleft()
        for j in graph.indices[graph.indptr[i]:graph.indptr[i + 1]]:
            if not supported[j]:
                supported[j] = True
                queue.append(int(j))


def _model_anchors(beads, mesh):
    anchored = np.zeros(len(beads), dtype=bool)
    if mesh is None or not len(beads):
        return anchored
    if len(beads) > 512:
        for start in range(0, len(beads), 512):
            anchored[start:start + 512] = _model_anchors(beads[start:start + 512], mesh)
        return anchored
    xyz, radii = beads[:, :3], beads[:, 3] * 0.5
    # A downward ray must hit an upward face directly below the bead. A nearby
    # window wall or lower corner is not a foundation.
    directions = np.tile([0.0, 0.0, -1.0], (len(beads), 1))
    locations, rays, faces = mesh.ray.intersects_location(xyz, directions, multiple_hits=False)
    if not len(rays):
        return anchored
    normals = mesh.face_normals[faces]
    height = xyz[rays, 2] - locations[:, 2]
    tolerance = np.maximum(1e-7, 0.02 * radii[rays])
    plane_distance = height * normals[:, 2]
    valid = ((normals[:, 2] >= math.sqrt(0.5) - 1e-9)
             & (np.abs(plane_distance - radii[rays]) <= tolerance))
    candidates = rays[valid]
    if len(candidates):
        # At an edge, the infinite plane can touch the sphere while the actual
        # triangle does not. Verify contact with the finite original surface.
        closest, distance, triangles = mesh.nearest.on_surface(xyz[candidates])
        normal = mesh.face_normals[triangles]
        delta = xyz[candidates] - closest
        tol = np.maximum(1e-7, 0.02 * radii[candidates])
        valid = ((normal[:, 2] >= math.sqrt(0.5) - 1e-9)
                 & (delta[:, 2] > 0)
                 & (np.abs(distance - radii[candidates]) <= tol)
                 & (np.einsum("ij,ij->i", delta, normal) >= radii[candidates] - tol))
        anchored[candidates[valid]] = True
    return anchored


def _initial_support(beads, mesh, bed_z, graph):
    radii = beads[:, 3] * 0.5
    bottom = beads[:, 2] - radii
    tolerance = np.maximum(1e-7, 0.02 * radii)
    bed = ((bottom <= bed_z + tolerance) & (beads[:, 2] >= bed_z)
           & (bottom >= bed_z - 0.25 * radii))
    # A true model foot remains an anchor even when a lower, floating neighbour
    # happens to touch it. Such a neighbour must not invalidate the real foot.
    roots = np.flatnonzero(np.asarray(graph.sum(axis=0)).ravel() == 0)
    supported = np.zeros(len(beads), dtype=bool)
    _grow_supported(graph, supported, np.flatnonzero(bed))
    candidates = np.flatnonzero(~supported)
    model = _model_anchors(beads[candidates], mesh)
    _grow_supported(graph, supported, candidates[model])
    return supported, roots


def supported_mask(seeds, mesh, bed_z=0.0):
    """Return beads reachable from real foundations using upward edges only."""
    if not len(seeds):
        return np.zeros(0, dtype=bool)
    beads = np.asarray(seeds, dtype=float)
    _, graph = _graph(beads)
    return _initial_support(beads, mesh, bed_z, graph)[0]


def repair_support_paths(seeds, mesh, bed_z, bead_diameter, xy_clearance,
                         z_gap=0.3, max_beads=None, allow_model=True,
                         progress_callback=None):
    """Add short downward paths before pruning any unsupported original beads.

    Reuse a nearby supported branch when a printable straight segment exists;
    otherwise extend vertically to the actual model top or bed. Every added
    sphere is checked against the original mesh, including the landing foot.
    Unroutable beads are left for the final bottom-up pruning pass.
    """
    if not len(seeds):
        return list(seeds), 0
    beads = np.asarray(seeds, dtype=float)
    tree, graph = _graph(beads)
    supported, roots = _initial_support(beads, mesh if allow_model else None, bed_z, graph)
    added = []
    radius = 0.5 * bead_diameter
    step = 0.88 * min(bead_diameter, float(beads[:, 3].min()))
    max_radius = max(radius, beads[:, 3].max() * 0.5)
    budget = max_beads if max_beads is not None else 2_000_000
    pq = mesh.nearest if mesh is not None else None
    added_tree = None
    indexed_added = 0
    point_safety = {}

    def check_points(points):
        result = np.ones(len(points), dtype=bool)
        if pq is None:
            return result
        for start in range(0, len(points), 256):
            p = points[start:start + 256]
            closest, distance, faces = pq.on_surface(p)
            delta = closest - p
            normals = mesh.face_normals[faces]
            below = ((delta[:, 2] < 0) & (normals[:, 2] >= math.sqrt(0.5) - 1e-9)
                     & (np.abs(delta[:, 2]) >= 0.7 * distance))
            above = (delta[:, 2] > 0) & (delta[:, 2] >= 0.7 * distance)
            required = np.full(len(p), radius + xy_clearance)
            required[above] = radius + min(z_gap, xy_clearance)
            required[below] = radius
            result[start:start + len(p)] = ((distance >= required - 1e-7)
                                            & (pq.signed_distance(p) <= 1e-7))
        return result

    def safe_path(path):
        keys = [tuple(point[:3]) for point in path]
        missing = list(dict.fromkeys(key for key in keys if key not in point_safety))
        if missing:
            point_safety.update(zip(missing, check_points(np.asarray(missing))))
        return all(point_safety[key] for key in keys)

    def chain(top, bottom, existing):
        length = np.linalg.norm(top - bottom)
        count = max(1, int(math.ceil(length / step)))
        if len(seeds) + len(added) + count > budget:
            return None
        # Exclude the existing top, and also exclude a reused lower bead.
        stops = count if not existing else count - 1
        return [(*map(float, top + (bottom - top) * (i / count)), bead_diameter)
                for i in range(1, stops + 1)]

    roots = roots[~supported[roots]]
    # Ray casting one foot at a time spends more time in geometry setup than in
    # the actual query. Precompute possible vertical foundations in batches.
    floors = beads[roots, :3].copy()
    floors[:, 2] = bed_z + 0.8 * radius
    hit_model = np.zeros(len(roots), dtype=bool)
    if mesh is not None:
        for start in range(0, len(roots), 512):
            batch = beads[roots[start:start + 512], :3]
            loc, ray, face = mesh.ray.intersects_location(
                batch, np.tile([0.0, 0.0, -1.0], (len(batch), 1)), multiple_hits=False)
            indices = start + ray
            hit_model[indices] = True
            normal_z = mesh.face_normals[face, 2]
            valid = (normal_z >= math.sqrt(0.5) - 1e-9) & allow_model
            floors[indices[~valid], 2] = np.nan
            floors[indices[valid], 2] = loc[valid, 2] + radius / normal_z[valid]
        check = np.flatnonzero(hit_model & np.isfinite(floors[:, 2]))
        feet = np.column_stack((floors[check], np.full(len(check), bead_diameter)))
        floors[check[~_model_anchors(feet, mesh)], 2] = np.nan
    floor_by_root = dict(zip(roots, floors))
    # Most breaks are a few missing bead layers at the model foot. Check these
    # short vertical paths together, retaining exact point results for reuse.
    short_points = []
    for root, floor in zip(roots, floors):
        distance = beads[root, 2] - floor[2]
        if 1e-8 < distance <= 3 * bead_diameter:
            path = chain(beads[root, :3], floor, False)
            if path:
                short_points.extend(tuple(point[:3]) for point in path)
    if short_points:
        unique = list(dict.fromkeys(short_points))
        point_safety.update(zip(unique, check_points(np.asarray(unique))))

    for processed, root in enumerate(roots[np.argsort(beads[roots, 2], kind="stable")], 1):
        if progress_callback and (processed == 1 or processed % 100 == 0):
            progress_callback(f"아래 받침 연결 보완 · {processed}/{len(roots)}")
        if supported[root]:
            continue
        top = beads[root, :3]
        floor = floor_by_root[root]
        floor_distance = top[2] - floor[2]
        direct_path = None
        if 1e-8 < floor_distance <= 3 * bead_diameter:
            direct_path = chain(top, floor, False)
            if direct_path is not None and not safe_path(direct_path):
                direct_path = None
        direct_foot = direct_path is not None
        options = []
        # Search a bounded neighbourhood: reconnecting a short break should not
        # create a full new column when a lower printed branch is already close.
        nearby = []
        if not direct_foot:
            _, nearby = tree.query(top, k=min(128, len(beads)), distance_upper_bound=bead_diameter * 20)
        for j in np.atleast_1d(nearby):
            if j >= len(beads):
                continue
            dz = top[2] - beads[j, 2]
            if supported[j] and dz > 1e-8 and np.linalg.norm(top[:2] - beads[j, :2]) <= dz:
                options.append((float(np.linalg.norm(top - beads[j, :3])), beads[j, :3], True))
        # Reuse newly added stems too. Rebuild this small index in batches;
        # rebuilding the full original graph for every repaired foot is costly.
        if not direct_foot and len(added) - indexed_added >= 128:
            added_tree = cKDTree(np.asarray(added)[:, :3])
            indexed_added = len(added)
        new_candidates = [] if direct_foot else list(range(indexed_added, len(added)))
        if not direct_foot and added_tree is not None:
            _, nearby_added = added_tree.query(
                top, k=min(128, indexed_added), distance_upper_bound=bead_diameter * 20)
            new_candidates.extend(int(j) for j in np.atleast_1d(nearby_added) if j < indexed_added)
        for j in new_candidates:
            point = np.asarray(added[j][:3])
            dz = top[2] - point[2]
            length = np.linalg.norm(top - point)
            if dz > 1e-8 and length <= bead_diameter * 20 and np.linalg.norm(top[:2] - point[:2]) <= dz:
                options.append((float(length), point, True))
        options.sort(key=lambda item: item[0])
        options = options[:8]
        if np.isfinite(floor[2]) and floor[2] < top[2] - 1e-8:
            options.append((float(top[2] - floor[2]), floor, False))
        options.sort(key=lambda item: item[0])
        for _, bottom, existing in options:
            path = chain(top, bottom, existing)
            if path is None or not safe_path(path):
                continue
            added.extend(path)
            _grow_supported(graph, supported, [root])
            # A new path can also support neighbouring original beads, allowing
            # later roots to reuse it through those beads without duplicate stems.
            for point in path:
                for j in tree.query_ball_point(point[:3], 1.96 * max_radius):
                    delta = beads[j, :3] - point[:3]
                    if (not supported[j] and delta[2] > 1e-8
                            and np.linalg.norm(delta[:2]) <= delta[2] + 1e-9
                            and np.linalg.norm(delta) < 0.98 * (radius + beads[j, 3] * 0.5)):
                        _grow_supported(graph, supported, [j])
            break
    return list(seeds) + added, len(added)
