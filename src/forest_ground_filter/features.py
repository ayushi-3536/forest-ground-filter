"""Geometry helpers used by the ground filters.

This covers thinning the cloud into voxels, measuring how high each point sits above the local floor,
and describing the shape of each point's neighbourhood (line-like, plane-like or volume-like).

Please don't change these lightly: they are exactly the code our benchmark numbers come from, so any edit
here changes the results.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

KNN = 96  # at most this many neighbours are used for each shape estimate


def voxel_first(xyz, size):
    """Keep one point per voxel of edge length `size`.

    Returns the indices of the kept points, and for every input point the index of the voxel it fell into,
    so labels computed on the kept points can be copied back to all points afterwards.
    """
    v = np.floor((xyz - xyz.min(0)) / size).astype(np.int64)  # which voxel (i, j, k) each point is in
    # Turn (i, j, k) into one integer so np.unique can group points by voxel. The three large primes are the
    # usual spatial-hash constants (Teschner et al. 2003); they scramble the coordinates so different voxels
    # almost never share a key. A clash is possible but extremely rare, and kept as is for reproducibility.
    key = (v[:, 0] * 73856093) ^ (v[:, 1] * 19349663) ^ (v[:, 2] * 83492791)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first, inv


def cell_index(xy, cell):
    """Which square column (seen from above, `cell` metres wide) each point belongs to, as one integer."""
    ij = np.floor((xy - xy.min(0)) / cell).astype(np.int64)
    return ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]


def ground_surface(xyz, cell=2.0):
    """How high each point is above the local floor, and which way the terrain faces there.

    The cloud is split into 2 m columns. The lowest point of each column is taken as its floor, and a
    point's height is simply its z minus that floor. Treating the column floors as a coarse terrain model
    also gives a slope, which we return as a normal vector per point.
    """
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    ni, nj = ij.max(0) + 1
    flat = ij[:, 0] * nj + ij[:, 1]
    zmin = np.full(ni * nj, np.inf)
    np.minimum.at(zmin, flat, xyz[:, 2])
    h = xyz[:, 2] - zmin[flat]
    Z = zmin.reshape(ni, nj)
    empty = ~np.isfinite(Z)
    if empty.any():  # columns without points borrow the floor of their nearest neighbour
        idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
        Z = Z[tuple(idx)]
    gx, gy = np.gradient(Z, cell) if min(ni, nj) > 1 else (np.zeros_like(Z), np.zeros_like(Z))
    n = np.stack([-gx, -gy, np.ones_like(Z)], -1).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return h, n[flat]


def planarity(cloud, query, r, k=KNN, chunk=100_000):
    """How flat the neighbourhood of each query point is.

    We fit an ellipsoid (PCA) to the neighbours within radius `r`. If they spread out along a sheet, the
    two long axes are similar and the third is short, so planarity (s2 - s3) / s1 is high. We also return
    the direction of the short axis (the surface normal) and how many neighbours were found.
    """
    tree = cKDTree(cloud)
    n = len(query)
    plan = np.zeros(n, np.float32)
    normal = np.zeros((n, 3), np.float32)
    cnt = np.zeros(n, np.int32)
    k = min(k, len(cloud))
    for s in range(0, n, chunk):
        d, i = tree.query(query[s:s + chunk], k=k, distance_upper_bound=r, workers=1)
        if d.ndim == 1:
            d, i = d[:, None], i[:, None]
        m = np.isfinite(d)
        i = np.where(m, i, 0)
        nb = cloud[i]
        c_ = m.sum(1)
        w = m[..., None]
        mu = (nb * w).sum(1) / np.maximum(c_, 1)[:, None]
        dc = (nb - mu[:, None]) * w
        cov = np.einsum("nki,nkj->nij", dc, dc) / np.maximum(c_, 1)[:, None, None]
        cov += np.eye(3) * 1e-9
        ev, evec = np.linalg.eigh(cov)  # eigenvalues come out smallest first
        e = np.sqrt(np.clip(ev, 0, None))[:, ::-1]  # flip so the longest axis comes first
        plan[s:s + chunk] = (e[:, 1] - e[:, 2]) / np.maximum(e[:, 0], 1e-8)
        normal[s:s + chunk] = evec[:, :, 0]  # the shortest axis points out of the surface
        cnt[s:s + chunk] = c_
    return plan, normal, cnt


def scale_clouds(v, scales, grid):
    """Prepare one thinned copy of the cloud per radius, plus a search tree for it.

    Large radii don't need every point, so each copy keeps one point per (radius / 6) voxel. That keeps
    the large-radius shape estimates fast without changing what they measure.
    """
    out = {}
    for s in scales:
        g = max(s / 6, grid)
        c = v if g <= grid else v[voxel_first(v, g)[0]]
        out[s] = (c, cKDTree(c))
    return out


def multiscale(clouds, scales, query, k=KNN, count_scale=1.0):
    """CANUPO's shape fingerprint for each query point (Brodu & Lague 2012).

    At every radius we ask two questions about the neighbourhood: how line-like is it,
    a1D = (s1 - s2) / s1, and how plane-like, a2D = (s2 - s3) / s1. Ten radii give 20 numbers per point.
    Ground, for example, is rough up close but flat further out, while a stem looks like a line up close.

    Also returns the number of neighbours at `count_scale`, so callers can spot points that are too
    isolated to describe.
    """
    n = len(query)
    feat = np.zeros((n, 2 * len(scales)), np.float32)
    cnt1 = np.zeros(n, np.int32)
    for j, s in enumerate(scales):
        cloud, tree = clouds[s]
        kk = min(k, len(cloud))
        for a in range(0, n, 100_000):
            d, i = tree.query(query[a:a + 100_000], k=kk, distance_upper_bound=s, workers=1)
            if d.ndim == 1:
                d, i = d[:, None], i[:, None]
            m = np.isfinite(d)
            i = np.where(m, i, 0)
            nb = cloud[i]
            c = m.sum(1)
            w = m[..., None]
            mu = (nb * w).sum(1) / np.maximum(c, 1)[:, None]
            dc = (nb - mu[:, None]) * w
            cov = np.einsum("nki,nkj->nij", dc, dc) / np.maximum(c, 1)[:, None, None] + np.eye(3) * 1e-12
            e = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0, None))[:, ::-1]
            e1 = np.maximum(e[:, 0], 1e-9)
            a1, a2 = (e[:, 0] - e[:, 1]) / e1, (e[:, 1] - e[:, 2]) / e1
            few = c < 3  # too few neighbours to fit a shape; call it a blob (neither line nor plane)
            a1[few], a2[few] = 0.0, 0.0
            feat[a:a + 100_000, 2 * j], feat[a:a + 100_000, 2 * j + 1] = a1, a2
            if s == count_scale:
                cnt1[a:a + 100_000] = c
    return feat, cnt1


def lda_fit(xg, xv, ridge=1e-4):
    """Learn a simple linear rule that separates ground examples from vegetation examples.

    This is Fisher's linear discriminant: it finds the weighted sum of the features where the two groups
    lie furthest apart, oriented so ground scores high. We then pick the cut-off on that score that gets
    both groups right as often as possible.

    Returns the weights, the cut-off, and how well the examples themselves are separated (0.5 to 1).
    """
    mg, mv = xg.mean(0), xv.mean(0)
    sw = np.cov(xg, rowvar=False) + np.cov(xv, rowvar=False)
    sw += ridge * np.trace(sw) / len(sw) * np.eye(len(sw))
    w = np.linalg.solve(sw, mg - mv)
    pg, pv = xg @ w, xv @ w
    cand = np.unique(np.quantile(np.concatenate([pg, pv]), np.linspace(0.01, 0.99, 197)))
    bacc = [(np.mean(pg >= t) + np.mean(pv < t)) / 2 for t in cand]
    t = cand[int(np.argmax(bacc))]
    return w, t, float(max(bacc))
