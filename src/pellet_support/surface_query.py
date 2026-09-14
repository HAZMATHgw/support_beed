"""Nearest-surface queries that reuse the index of an unchanged mesh."""

import numpy as np
from scipy.spatial import cKDTree
from trimesh.constants import tol
from trimesh.triangles import closest_point
from trimesh.util import diagonal_dot


class CachedSurfaceQuery:
    """Keep vertex/triangle indices only for one operation on a fixed mesh.

    Trimesh's proximity query rebuilds its referenced-vertex KD-tree for every
    batch. Path repair needs thousands of small batches on the same geometry.
    The nearest vertex supplies the same conservative candidate bounds here;
    exact triangle distances and Trimesh's two-face tie rule select the result.
    The caller must discard this object before changing the model geometry.
    """

    def __init__(self, mesh):
        self.vertices = cKDTree(mesh.vertices[mesh.referenced_vertices])
        self.bounds = mesh.triangles_tree
        self.triangles = np.asarray(mesh.triangles)
        self.normals = np.asarray(mesh.face_normals)

    def on_surface(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (n,3)")
        if not len(points):
            return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.intp)
        reach = self.vertices.query(points)[0].reshape((-1, 1)) + tol.merge
        lower, upper = points - reach, points + reach
        try:
            candidates, sizes = self.bounds.intersection_v(lower, upper)
            candidates = np.asarray(candidates, dtype=np.intp)
            sizes = np.asarray(sizes, dtype=np.intp)
        except Exception:
            groups = [list(self.bounds.intersection(bound))
                      for bound in np.column_stack((lower, upper))]
            sizes = np.asarray([len(group) for group in groups], dtype=np.intp)
            candidates = np.concatenate(groups).astype(np.intp, copy=False)
        repeated = np.repeat(np.arange(len(points)), sizes)
        nearest = closest_point(self.triangles[candidates], points[repeated])
        delta = points[repeated] - nearest
        squared = diagonal_dot(delta, delta)
        boundaries = np.cumsum(sizes)[:-1]
        # Preserve candidate ordering even in exact ties at shared edges.
        best = np.asarray([group.argsort()[:2] if len(group) > 1 else [0, 0]
                           for group in np.split(squared, boundaries)], dtype=np.intp)
        best[1:] += boundaries[:, None]
        pair_distances = squared[best]
        selected = best[:, 0].copy()
        ambiguous = ((np.ptp(pair_distances, axis=1) < tol.merge)
                     & np.all(np.abs(pair_distances) > tol.merge, axis=1))
        if ambiguous.any():
            pair = best[ambiguous]
            normals = self.normals[candidates[pair]]
            directions = delta[pair] / np.sqrt(pair_distances[ambiguous])[:, :, None]
            preference = (normals * directions).sum(axis=2).argmax(axis=1)
            selected[ambiguous] = pair[np.arange(len(pair)), preference]
        return nearest[selected], np.sqrt(squared[selected]), candidates[selected]
