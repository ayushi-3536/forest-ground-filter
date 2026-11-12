"""Geometry shared by the filters: voxelisation, height above the local floor, local PCA descriptors.

These functions are kept identical to the code the published numbers were produced with, so results are
reproducible bit for bit.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

KNN = 96  # neighbours used per PCA (inside the radius)


def voxel_first(xyz, size):
    """First point of every `size` voxel. Returns (index of the kept points, voxel index of every point)."""
    v = np.floor((xyz - xyz.min(0)) / size).astype(np.int64)
    key = (v[:, 0] * 73856093) ^ (v[:, 1] * 19349663) ^ (v[:, 2] * 83492791)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first, inv


def cell_index(xy, cell):
    """Flat index of the `cell` x `cell` m column of every point (origin at the cloud's xy minimum)."""
    ij = np.floor((xy - xy.min(0)) / cell).astype(np.int64)
    return ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]


def ground_surface(xyz, cell=2.0):
    """Height above the lowest point of the point's 2 m cell, and the terrain normal of that minimum surface."""
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    ni, nj = ij.max(0) + 1
    flat = ij[:, 0] * nj + ij[:, 1]
    zmin = np.full(ni * nj, np.inf)
    np.minimum.at(zmin, flat, xyz[:, 2])
    h = xyz[:, 2] - zmin[flat]
    Z = zmin.reshape(ni, nj)
    empty = ~np.isfinite(Z)
    if empty.any():  # fill empty cells from the nearest occupied one
        idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
        Z = Z[tuple(idx)]
    gx, gy = np.gradient(Z, cell) if min(ni, nj) > 1 else (np.zeros_like(Z), np.zeros_like(Z))
    n = np.stack([-gx, -gy, np.ones_like(Z)], -1).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return h, n[flat]


def planarity(cloud, query, r, k=KNN, chunk=100_000):
    """Planarity a2D = (s2 - s3) / s1, unit normal and neighbour count per query point, neighbours within r."""
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
        ev, evec = np.linalg.eigh(cov)  # ascending
        e = np.sqrt(np.clip(ev, 0, None))[:, ::-1]  # s1 >= s2 >= s3
        plan[s:s + chunk] = (e[:, 1] - e[:, 2]) / np.maximum(e[:, 0], 1e-8)
        normal[s:s + chunk] = evec[:, :, 0]  # smallest eigenvector = surface normal
        cnt[s:s + chunk] = c_
    return plan, normal, cnt


def scale_clouds(v, scales, grid):
    """Per scale s: a (s / 6)-voxel copy of the cloud (never finer than `grid`) and its KD-tree."""
    out = {}
    for s in scales:
        g = max(s / 6, grid)
        c = v if g <= grid else v[voxel_first(v, g)[0]]
        out[s] = (c, cKDTree(c))
    return out


def multiscale(clouds, scales, query, k=KNN, count_scale=1.0):
    """CANUPO descriptor: barycentric (a1D, a2D) at every scale -> (n, 2 * len(scales)); and the neighbour
    count at `count_scale`. a1D = (s1 - s2) / s1, a2D = (s2 - s3) / s1 (Brodu & Lague 2012)."""
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
            few = c < 3  # undefined PCA: treat as isotropic (3D)
            a1[few], a2[few] = 0.0, 0.0
            feat[a:a + 100_000, 2 * j], feat[a:a + 100_000, 2 * j + 1] = a1, a2
            if s == count_scale:
                cnt1[a:a + 100_000] = c
    return feat, cnt1


def lda_fit(xg, xv, ridge=1e-4):
    """Fisher LDA direction (ground projects high) and the threshold that best separates the two sample sets
    (balanced accuracy). Returns (w, t, balanced accuracy)."""
    mg, mv = xg.mean(0), xv.mean(0)
    sw = np.cov(xg, rowvar=False) + np.cov(xv, rowvar=False)
    sw += ridge * np.trace(sw) / len(sw) * np.eye(len(sw))
    w = np.linalg.solve(sw, mg - mv)
    pg, pv = xg @ w, xv @ w
    cand = np.unique(np.quantile(np.concatenate([pg, pv]), np.linspace(0.01, 0.99, 197)))
    bacc = [(np.mean(pg >= t) + np.mean(pv < t)) / 2 for t in cand]
    t = cand[int(np.argmax(bacc))]
    return w, t, float(max(bacc))
