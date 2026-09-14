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


def _outside_from_surface(mesh, points, closest, distance, faces):
    """Apply Trimesh's signed-distance test to an existing nearest query.

    Face-interior projections determine the sign from the outward normal;
    projections beyond an edge still need the same containment ray test.
    Keeping that distinction matters around concave edges and open meshes.
    """
    from trimesh.constants import tol
    from trimesh.triangles import points_to_barycentric
    from trimesh.util import diagonal_dot

    outside = distance <= 1e-7
    nonzero = np.flatnonzero(distance > tol.merge)
    if not len(nonzero):
        return outside
    normals = mesh.face_normals[faces[nonzero]]
    delta = points[nonzero] - closest[nonzero]
    projection = points[nonzero] - (normals.T * diagonal_dot(delta, normals)).T
    barycentric = points_to_barycentric(mesh.triangles[faces[nonzero]], projection)
    on_face = ~((barycentric < -tol.merge) | (barycentric > 1 + tol.merge)).any(axis=1)
    direct = nonzero[on_face]
    sign = np.sign(diagonal_dot(normals[on_face], points[direct] - projection[on_face]))
    outside[direct] = -distance[direct] * sign <= 1e-7
    edge = nonzero[~on_face]
    if len(edge):
        inside = mesh.ray.contains_points(points[edge])
        outside[edge] = distance[edge] * (inside.astype(int) * 2 - 1) <= 1e-7
    return outside


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
    from .surface_query import CachedSurfaceQuery

    pq = CachedSurfaceQuery(mesh) if mesh is not None else None
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
            clear = distance >= required - 1e-7
            # A point already too close cannot become safe after a sign test.
            # Reuse the nearest face for the remaining points instead of doing
            # a second full nearest-triangle query inside signed_distance.
            check = np.flatnonzero(clear)
            if len(check):
                clear[check] &= _outside_from_surface(
                    mesh, p[check], closest[check], distance[check], faces[check])
            result[start:start + len(p)] = clear
        return result

    def safe_path(path):
        keys = [tuple(point[:3]) for point in path]
        if any(key in point_safety and not point_safety[key] for key in keys):
            return False
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


def close_ceiling_gaps(seeds, mesh, targets, bead_diameter, xy_clearance,
                       embed=0.0, max_beads=None, contact_offset=None,
                       contact_status=None, progress_callback=None,
                       contact_origins=None, contact_positions=None):
    """Connect leaves to a real underside without moving their foundations.

    Try the nearest underside and the first ceiling directly above the leaf.
    Every extension must rise at most 45 degrees from vertical and overlap its
    lower bead, including mixed diameters. Original beads are never lifted.
    XY clearance applies to the approach. The final contact uses exact sphere
    clearance so narrow roofs can be held up; every neighbouring face is still
    checked against penetration, including side walls and thin roof tops.

    Returns (seeds, added_count, resolved_z), preserving the public tuple from
    the angled-contact correction. Optional contact_positions records full XYZ
    coordinates for cleanup/diagnostics when the corrected tip moves sideways.
    contact_origins identifies the intended surviving leaf, including contacts
    clamped to the bed; contact_offset is a fallback for older callers.
    """
    resolved = [None] * len(targets)
    if contact_status is not None:
        contact_status[:] = [False] * len(targets)
    if contact_positions is not None:
        contact_positions[:] = resolved
    if not len(seeds) or not len(targets) or mesh is None:
        return list(seeds), 0, resolved
    from trimesh.triangles import closest_point

    beads = np.asarray(seeds, dtype=float)
    radius = 0.5 * bead_diameter
    embed = min(max(0.0, embed), 0.5 * radius)
    budget = max_beads if max_beads is not None else 2_000_000
    xy_tree = cKDTree(beads[:, :2])
    face_tree = mesh.triangles_tree
    triangles, normals = mesh.triangles, mesh.face_normals
    ceiling_normal = 0.3
    added, selected = [], []
    completed = {}
    for target_index, (x, y, expected_z) in enumerate(targets):
        nearby = xy_tree.query_ball_point([x, y], bead_diameter * 2.0)
        nearby = [j for j in nearby if beads[j, 2] < expected_z - 1e-8]
        nominal = None
        if contact_origins is not None:
            nominal = np.asarray(contact_origins[target_index], dtype=float)
        elif contact_offset is not None:
            nominal = np.array([x, y, expected_z - contact_offset])
        if nominal is not None:
            nearby = [j for j in nearby
                      if np.linalg.norm(beads[j, :3] - nominal) <= 2 * bead_diameter]
            key = lambda j: np.linalg.norm(beads[j, :3] - nominal)
        else:
            key = lambda j: (expected_z - beads[j, 2],
                             np.linalg.norm(beads[j, :2] - [x, y]))
        if nearby:
            selected.append((target_index, min(nearby, key=key), expected_z))
    if not selected:
        return list(seeds), 0, resolved

    top_ids = list(dict.fromkeys(item[1] for item in selected))
    ceilings, nearest, upward_hits = {}, {}, {}
    for start in range(0, len(top_ids), 512):
        ids = top_ids[start:start + 512]
        points = beads[ids, :3]
        locations, rays, faces = mesh.ray.intersects_location(
            points, np.tile([0.0, 0.0, 1.0], (len(ids), 1)), multiple_hits=True)
        for location, ray, face in zip(locations, rays, faces):
            upward_hits.setdefault(ids[ray], []).append((location, normals[face]))
        for idx in ids:
            hits = upward_hits.get(idx, [])
            hits.sort(key=lambda hit: hit[0][2])
            if hits and hits[0][1][2] <= -ceiling_normal:
                ceilings[idx] = hits[0]
        closest, distances, faces = mesh.nearest.on_surface(points)
        for idx, point, distance, face in zip(ids, closest, distances, faces):
            if normals[face, 2] <= -ceiling_normal and point[2] > beads[idx, 2]:
                nearest[idx] = (point, distance, normals[face])

    def safe_path(path, surface_point, surface_normal):
        points = np.asarray(path)[:, :3]
        point_ids, face_ids = [], []
        reach = radius + max(xy_clearance, 0.0) + 1e-7
        for i, point in enumerate(points):
            candidates = list(face_tree.intersection(
                np.concatenate((point - reach, point + reach))))
            point_ids.extend([i] * len(candidates))
            face_ids.extend(candidates)
        if not face_ids:
            return False
        point_ids = np.asarray(point_ids, dtype=int)
        face_ids = np.asarray(face_ids, dtype=int)
        final_contact = False
        for start in range(0, len(face_ids), 8192):
            ids = point_ids[start:start + 8192]
            faces = face_ids[start:start + 8192]
            points_for_faces = points[ids]
            closest = closest_point(triangles[faces], points_for_faces)
            delta = closest - points_for_faces
            distance = np.linalg.norm(delta, axis=1)
            normal = normals[faces]
            above = (normal[:, 2] <= -ceiling_normal) & (delta[:, 2] > 0)
            last = ids == len(points) - 1
            required = np.full(len(ids), radius + xy_clearance)
            # Opposite faces behind a thin roof/floor are not side walls.
            required[np.abs(normal[:, 2]) >= ceiling_normal] = radius
            # Horizontal clearance ends at the contact interface. Roof edges
            # behind that surface must not reject a valid underside joint;
            # their actual sphere clearance is still checked, including walls.
            behind_contact = ((closest - surface_point) @ surface_normal <= embed + 1e-7)
            required[behind_contact] = radius
            required[last] = radius
            required[above & last] = radius - embed
            if np.any(distance < required - 1e-7):
                return False
            final_contact |= bool(np.any(last & above & (distance <= radius + 1e-7)))
        return final_contact

    def mark(index, position):
        if position is not None:
            position = tuple(float(v) for v in position)
            resolved[index] = position[2]
            if contact_status is not None:
                contact_status[index] = True
            if contact_positions is not None:
                contact_positions[index] = position

    for processed, (target_index, top_idx, expected_z) in enumerate(selected, 1):
        if progress_callback and (processed == 1 or processed % 100 == 0):
            progress_callback(f"천장 틈 보정 · {processed}/{len(selected)}")
        top = beads[top_idx, :3]
        # Identifying a target may require seeing later intersections, but a
        # lower shelf cannot be reported as supporting the intended upper one.
        hits = upward_hits.get(top_idx, [])
        downward_hits = [hit for hit in hits if hit[1][2] <= -ceiling_normal]
        if downward_hits and hits:
            intended = min(downward_hits, key=lambda hit: abs(hit[0][2] - expected_z))
            if intended[0][2] > hits[0][0][2] + 1e-7:
                continue
        options = []
        near = nearest.get(top_idx)
        if near is not None and abs(near[0][2] - expected_z) <= 3 * bead_diameter:
            point, distance, normal = near
            original_radius = 0.5 * beads[top_idx, 3]
            # Embedding permits penetration, never an equally large air gap.
            if original_radius - embed - 1e-7 <= distance <= original_radius + 1e-7:
                mark(target_index, top)
                continue
            options.append((point + normal * (radius - embed), point, normal))
        ceiling = ceilings.get(top_idx)
        if ceiling is not None and abs(ceiling[0][2] - expected_z) <= 3 * bead_diameter:
            point, normal = ceiling
            options.append((np.array([top[0], top[1],
                                      point[2] - (radius - embed) / -normal[2]]), point, normal))
        if not options:
            continue
        if top_idx in completed:
            mark(target_index, completed[top_idx])
            continue
        options.sort(key=lambda item: np.linalg.norm(item[0] - top))
        for target, surface_point, surface_normal in options:
            delta = target - top
            length = np.linalg.norm(delta)
            if delta[2] <= 1e-8 or np.linalg.norm(delta[:2]) > delta[2] + 1e-9:
                continue
            first_step = 0.88 * (radius + 0.5 * beads[top_idx, 3])
            step = 0.88 * bead_diameter
            count = 1 + max(0, int(math.ceil((length - first_step) / step)))
            if len(seeds) + len(added) + count > budget:
                continue
            distances = ([length] if count == 1 else
                         np.linspace(first_step, length, count).tolist())
            path = [(*map(float, top + delta * (distance / length)), bead_diameter)
                    for distance in distances]
            if not safe_path(path, surface_point, surface_normal):
                continue
            added.extend(path)
            completed[top_idx] = target
            mark(target_index, target)
            break
    return list(seeds) + added, len(added), resolved


def dedupe_seeds(seeds, tolerance=0.02):
    """Remove identical spheres apart from numerical noise, retaining order.

    Even a small centre offset may provide the only overlap with a neighbour,
    and coincident spheres of different sizes need not share a foundation.
    Without rechecking model contact and every support edge, only identical
    geometry is redundant. ``tolerance`` remains accepted for compatibility;
    it no longer allows removal of geometrically distinct spheres.
    """
    if not len(seeds):
        return list(seeds), 0
    beads = np.asarray(seeds, dtype=float)
    # Sorting rows avoids materializing O(n^2) pairs when many construction
    # passes produce the same bead, while preserving every retained position.
    keys = beads.copy()
    # Symmetric paths can produce +5e-17 and -5e-17 for the same coordinate.
    # This precision is far below geometry tolerances; retained spheres are
    # not rounded or moved, and their diameters must still match exactly.
    keys[:, :3] = np.round(keys[:, :3], 12)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return [seeds[int(i)] for i in first], len(seeds) - len(first)


def prune_disconnected_fill(seeds, contact_points, bed_z, max_cluster_size=8,
                            touch_slack=0.02):
    """Drop small bead clusters that neither reach the bed nor hold up a contact.

    A trunk's hex-packed fill assumes each layer's disk lines up with its
    neighbours; a sharp bend on a thin, twisty branch can strand a handful of
    beads that end up spatially isolated from the rest of their own trunk.
    Each stray bead is individually printable (it can rest against the model
    or a neighbour), so the existing bed/model-anchor check does not remove
    it, but the cluster does not reach the bed and is not the reason any
    overhang is held up -- it is leftover clutter, not support. Only clusters
    at most ``max_cluster_size`` beads are considered, so a genuine large
    branch that legitimately lands away from a sampled contact is never
    touched.
    """
    if not len(seeds):
        return list(seeds), 0
    beads = np.asarray(seeds, dtype=float)
    radii = beads[:, 3] * 0.5
    tree = cKDTree(beads[:, :3])
    pairs = tree.query_pairs(float(radii.max()) * 2.0 * (1.0 + touch_slack),
                             output_type="ndarray")
    n = len(beads)
    if len(pairs):
        delta = beads[pairs[:, 1], :3] - beads[pairs[:, 0], :3]
        dist = np.linalg.norm(delta, axis=1)
        touching = dist <= (1.0 + touch_slack) * (radii[pairs[:, 0]] + radii[pairs[:, 1]])
        pairs = pairs[touching]
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    graph = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n)) if len(pairs) else csr_matrix((n, n), dtype=bool)
    component_count, labels = connected_components(graph, directed=False)
    bed_tolerance = np.maximum(1e-6, 0.25 * radii)
    on_bed = (beads[:, 2] - radii) <= bed_z + bed_tolerance
    sizes = np.bincount(labels, minlength=component_count)
    protected = ((sizes > max_cluster_size)
                 | (np.bincount(labels, weights=on_bed,
                                minlength=component_count) > 0))
    # Classify all remaining groups in one pass. Scanning all n labels once
    # per isolated cluster otherwise turns cleanup into quadratic work.
    candidates = np.flatnonzero(~protected[labels])
    if len(candidates) and len(contact_points):
        contact_tree = cKDTree(np.asarray(contact_points, dtype=float))
        near, _ = contact_tree.query(beads[candidates, :3])
        at_contact = candidates[near <= radii[candidates] * 1.5]
        protected[labels[at_contact]] = True
    keep = protected[labels]
    return [s for s, k in zip(seeds, keep) if k], int((~keep).sum())
