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


def close_ceiling_gaps(seeds, mesh, targets, bead_diameter, xy_clearance,
                       embed=0.0, max_beads=None):
    """Stack beads up to the real overhang surface under each leaf contact.

    A tree contact's height comes from one representative overhang sample,
    but wider contact spacing (used to cut bead counts) makes that sample
    less likely to match the true local ceiling height at the exact XY where
    the branch actually lands. The result is a leaf that stops short of the
    surface it is meant to hold up, sometimes by several millimetres.

    The support must actually touch the model to hold it up, so the fix
    lands the final bead ``embed`` into the surface (the same idea as
    neighbouring beads pressing into each other) rather than short of it —
    a real, if shallow, joint instead of a gap with nothing to bridge. This
    also closes the small by-design clearance every contact is placed with
    (not just the occasional large miss): when the shortfall fits within one
    bead's reach *and* lifting the existing top bead would not pull it out
    of reach of whatever it was already resting on, that bead is moved in
    place rather than stacking a near-duplicate on top of it. Otherwise
    (including when a "safe" lift would strand the bead beneath it) a chain
    of new beads is inserted above the original, unmoved top bead instead,
    so the connection that was already there never breaks.

    ``targets`` is a sequence of ``(x, y, expected_z)`` — the contact's
    original (pre-offset) overhang height, used only as a sanity check on
    the search; the real surface is found as the closest point on the mesh
    to the branch's own topmost bead, in whatever direction that is — not
    necessarily straight up, since a twisty branch need not lean the same
    way as its own ceiling.

    Returns ``(seeds, corrections, resolved_z)``. ``resolved_z`` has one
    entry per target: the z of the bead now sitting at (or past) that
    contact's true, embedded surface, or ``None`` where no correction could
    be made (no nearby bead, no usable nearby surface, budget exhausted,
    ...). Callers that
    check "does every contact have a connected bead" must compare against
    this corrected height instead of the contact's original, pre-correction
    height — a large but legitimate correction (the whole reason this
    function exists) would otherwise look identical to a real disconnection.
    """
    if not len(seeds) or not len(targets) or mesh is None:
        return list(seeds), 0, [None] * len(targets)
    beads = np.asarray(seeds, dtype=float)
    tree = cKDTree(beads[:, :2])
    radius = 0.5 * bead_diameter
    step = 0.88 * bead_diameter
    budget = max_beads if max_beads is not None else 2_000_000
    pq = mesh.nearest
    added = []
    moved = {}
    resolved: list = [None] * len(targets)
    point_safety = {}

    def check_points(points):
        result = np.ones(len(points), dtype=bool)
        required = radius + xy_clearance
        for start in range(0, len(points), 256):
            p = points[start:start + 256]
            distance = pq.on_surface(p)[1]
            result[start:start + len(p)] = ((distance >= required - 1e-7)
                                            & (pq.signed_distance(p) <= 1e-7))
        return result

    def safe_path(path):
        keys = [tuple(point[:3]) for point in path]
        missing = list(dict.fromkeys(key for key in keys if key not in point_safety))
        if missing:
            point_safety.update(zip(missing, check_points(np.asarray(missing))))
        return all(point_safety[key] for key in keys)

    for t_idx, (x, y, expected_z) in enumerate(targets):
        nearby = tree.query_ball_point([x, y], bead_diameter * 2.0)
        if not nearby:
            continue
        # A tall, densely routed model can put an unrelated branch's bead at
        # a similar XY but a wildly different Z (e.g. one lattice column
        # passing near another's base). Restrict to beads actually near this
        # contact's own expected height before taking "the top one", or a
        # distant branch gets mistaken for this contact's tip and the real
        # gap here never gets corrected.
        near_height = [j for j in nearby if beads[j, 2] <= expected_z + bead_diameter]
        if not near_height:
            continue
        top_idx = max(near_height, key=lambda j: beads[j, 2])
        top = beads[top_idx, :3]
        # The bead this branch already rests on, if any — lifting ``top``
        # must not pull it out of reach of this one, or the chain breaks
        # exactly where it used to be connected.
        below = [j for j in near_height if j != top_idx]
        below_idx = max(below, key=lambda j: beads[j, 2]) if below else None
        # A twisty branch does not necessarily lean the same way its actual
        # ceiling does, so the nearest overhang surface to a tip is often not
        # straight above it — a purely vertical ray can sail past a ceiling
        # that is off to one side and find nothing (or the wrong thing).
        # ``on_surface`` finds the true closest point in any direction, which
        # is what the printed gap actually depends on.
        closest, distance, face_idx = pq.on_surface([top])
        closest, distance, face_idx = closest[0], float(distance[0]), face_idx[0]
        normal = mesh.face_normals[face_idx]
        # Only a downward-facing surface is a ceiling this contact can hang
        # from; the closest point on a side wall or something below is not
        # what this correction is for. This does not need to match the 45
        # degree overhang threshold that flagged the contact in the first
        # place -- detection measures the angle from stacked cross-sections,
        # not one triangle's own normal, so a genuinely overhanging spot can
        # still land on a near-vertical triangle right at that threshold.
        # Reject only surfaces that plainly face sideways or upward.
        if normal[2] > -0.3:
            continue
        # The intended ceiling is close to the detection sample's original
        # height; a much farther point is unrelated geometry entirely.
        if abs(closest[2] - expected_z) > 3.0 * bead_diameter:
            continue
        gap = distance - radius + embed
        if gap <= 1e-9:
            resolved[t_idx] = float(top[2])
            continue
        direction = (closest - top) / distance if distance > 1e-9 else np.array([0.0, 0.0, 1.0])
        target = top + direction * gap
        target_centre = tuple(float(v) for v in target)
        safe_to_move = True
        if gap <= step and below_idx is not None:
            required = 0.98 * (radius + 0.5 * float(beads[below_idx, 3]))
            safe_to_move = np.linalg.norm(target - beads[below_idx, :3]) <= required
        if gap <= step and safe_to_move:
            # The base placement left only the usual by-design clearance
            # short of the surface. Lift that same bead into a shallow
            # embed rather than stacking a near-duplicate bead a fraction
            # of a millimetre above it.
            beads[top_idx, :3] = target
            moved[top_idx] = target_centre
            resolved[t_idx] = target_centre[2]
            continue
        count = max(1, int(math.ceil(gap / step)))
        if len(seeds) + len(added) + count > budget:
            continue
        path = [(*map(float, top + direction * (gap * (i / count))), bead_diameter)
                for i in range(1, count + 1)]
        # The last bead is meant to press into the surface by design, so
        # only the beads leading up to it are checked for stray collisions.
        path, final = path[:-1], path[-1:]
        if path and not safe_path(path):
            continue
        added.extend(path)
        added.extend(final)
        resolved[t_idx] = final[0][2]
    result = [(*moved[i], d) if i in moved else (x, y, z, d)
              for i, (x, y, z, d) in enumerate(seeds)]
    return result + added, len(added) + len(moved), resolved


def dedupe_seeds(seeds, tolerance=0.02):
    """Merge beads whose centres coincide within ``tolerance`` mm.

    Independent passes (trunk hex-fill, bracing, repair, ceiling-gap
    correction) can each place a bead near the same spot without checking
    what the others already put there. Two beads on top of each other add no
    strength and read as a lumpy, uneven clump — keep the first and drop the
    rest.

    A looser, overlap-ratio-based version of this (dropping any pair deeper
    than some fraction of a diameter, not just near-exact coincidence) was
    tried and reverted: bracing and repair deliberately press a new bead
    deep into an existing one to guarantee a firm joint between two trunks
    that would otherwise stand separately, and that check could not tell a
    redundant clump from an intentional deep joint — it silently deleted the
    very beads ``test_bracing_ties_separate_trunks_together`` and the
    point-four-mm connectivity tests exist to require. Only exact
    coincidence is safe to remove generically.
    """
    if not seeds:
        return list(seeds), 0
    beads = np.asarray(seeds, dtype=float)
    tree = cKDTree(beads[:, :3])
    pairs = tree.query_pairs(tolerance, output_type="ndarray")
    if not len(pairs):
        return list(seeds), 0
    drop = np.zeros(len(beads), dtype=bool)
    for a, b in pairs:
        if not drop[a]:
            drop[int(b)] = True
    kept = [s for s, d in zip(seeds, drop) if not d]
    return kept, int(drop.sum())


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
    if not seeds:
        return list(seeds), 0
    beads = np.asarray(seeds, dtype=float)
    radii = beads[:, 3] * 0.5
    tree = cKDTree(beads[:, :3])
    pairs = tree.query_pairs(float(radii.max()) * 2.0, output_type="ndarray")
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
    _, labels = connected_components(graph, directed=False)
    bed_tolerance = np.maximum(1e-6, 0.25 * radii)
    on_bed = (beads[:, 2] - radii) <= bed_z + bed_tolerance
    contacts = np.asarray(contact_points, dtype=float) if len(contact_points) else None
    contact_tree = cKDTree(contacts) if contacts is not None and len(contacts) else None
    keep = np.ones(n, dtype=bool)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) > max_cluster_size or on_bed[members].any():
            continue
        if contact_tree is not None:
            near, _ = contact_tree.query(beads[members, :3])
            if (near <= radii[members] * 1.5).any():
                continue
        keep[members] = False
    return [s for s, k in zip(seeds, keep) if k], int((~keep).sum())
