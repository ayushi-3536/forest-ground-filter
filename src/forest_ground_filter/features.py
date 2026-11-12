"""Geometry helpers used by the ground filters.

- Thin a cloud into voxels.
- Measure each point's height above the local floor.
- Describe the shape of each point's neighbourhood (line, plane or blob).

Note: the benchmark numbers come from exactly this code; any change here
changes the results.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# At most this many neighbours are used for each shape estimate.
KNN = 96


def voxel_first(xyz, size):
    """Keep one point per voxel of edge length `size`.

    Returns:
        first: indices of the kept points.
        inv: voxel index of every input point, to copy labels computed on
            the kept points back to all points.
    """
    # Which voxel (i, j, k) each point is in.
    v = np.floor((xyz - xyz.min(0)) / size).astype(np.int64)
    # Combine (i, j, k) into one integer key per voxel (spatial hash with the
    # usual large primes, Teschner et al. 2003). Different voxels almost
    # never share a key; the rare clash is kept for reproducibility.
    key = (v[:, 0] * 73856093) ^ (v[:, 1] * 19349663) ^ (v[:, 2] * 83492791)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first, inv


def cell_index(xy, cell):
    """Return the index of the `cell`-metre column (seen from above) of
    each point."""
    ij = np.floor((xy - xy.min(0)) / cell).astype(np.int64)
    return ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]


def ground_surface(xyz, cell=2.0):
    """Height above the local floor and terrain normal of each point.

    1. Split the cloud into `cell`-metre columns.
    2. Take the lowest point of each column as its floor.
    3. Height = z minus the floor of the point's column.
    4. Terrain normal = slope of the grid of column floors.
    """
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    ni, nj = ij.max(0) + 1
    flat = ij[:, 0] * nj + ij[:, 1]
    zmin = np.full(ni * nj, np.inf)
    np.minimum.at(zmin, flat, xyz[:, 2])
    h = xyz[:, 2] - zmin[flat]
    Z = zmin.reshape(ni, nj)
    empty = ~np.isfinite(Z)
    if empty.any():
        # Columns without points borrow the floor of their nearest neighbour.
        idx = distance_transform_edt(
            empty, return_distances=False, return_indices=True
        )
        Z = Z[tuple(idx)]
    if min(ni, nj) > 1:
        gx, gy = np.gradient(Z, cell)
    else:
        gx, gy = np.zeros_like(Z), np.zeros_like(Z)
    n = np.stack([-gx, -gy, np.ones_like(Z)], -1).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return h, n[flat]


def planarity(cloud, query, r, k=KNN, chunk=100_000):
    """Planarity, surface normal and neighbour count of each query point.

    1. Collect the neighbours within radius `r` (at most `k`).
    2. Fit an ellipsoid to them (PCA); axis lengths s1 >= s2 >= s3.
    3. Planarity = (s2 - s3) / s1: high when the neighbours form a sheet.
    4. Normal = direction of the shortest axis.
    """
    tree = cKDTree(cloud)
    n = len(query)
    plan = np.zeros(n, np.float32)
    normal = np.zeros((n, 3), np.float32)
    cnt = np.zeros(n, np.int32)
    k = min(k, len(cloud))
    for s in range(0, n, chunk):
        d, i = tree.query(
            query[s : s + chunk], k=k, distance_upper_bound=r, workers=1
        )
        if d.ndim == 1:
            d, i = d[:, None], i[:, None]
        m = np.isfinite(d)
        i = np.where(m, i, 0)
        nb = cloud[i]
        c_ = m.sum(1)
        w = m[..., None]
        mu = (nb * w).sum(1) / np.maximum(c_, 1)[:, None]
        dc = (nb - mu[:, None]) * w
        cov = np.einsum("nki,nkj->nij", dc, dc)
        cov = cov / np.maximum(c_, 1)[:, None, None]
        cov += np.eye(3) * 1e-9
        # eigh sorts smallest first; flip to s1 >= s2 >= s3.
        ev, evec = np.linalg.eigh(cov)
        e = np.sqrt(np.clip(ev, 0, None))[:, ::-1]
        plan[s : s + chunk] = (e[:, 1] - e[:, 2]) / np.maximum(e[:, 0], 1e-8)
        # Shortest axis = surface normal.
        normal[s : s + chunk] = evec[:, :, 0]
        cnt[s : s + chunk] = c_
    return plan, normal, cnt


def scale_clouds(v, scales, grid):
    """Build one thinned copy of the cloud and its search tree per radius.

    Each copy keeps one point per (radius / 6) voxel, never finer than
    `grid`, to keep large-radius estimates fast.
    """
    out = {}
    for s in scales:
        g = max(s / 6, grid)
        c = v if g <= grid else v[voxel_first(v, g)[0]]
        out[s] = (c, cKDTree(c))
    return out


def multiscale(clouds, scales, query, k=KNN, count_scale=1.0):
    """CANUPO descriptor of each query point (Brodu & Lague 2012).

    1. For each radius in `scales`, fit an ellipsoid to the neighbours.
    2. Compute a1D = (s1 - s2) / s1 (line-like) and a2D = (s2 - s3) / s1
       (plane-like).
    3. Stack them: 2 values per radius, 20 for the default 10 radii.

    Returns:
        feat: (n, 2 * len(scales)) descriptor.
        cnt: neighbour count at `count_scale`, to flag isolated points.
    """
    n = len(query)
    feat = np.zeros((n, 2 * len(scales)), np.float32)
    cnt1 = np.zeros(n, np.int32)
    step = 100_000
    for j, s in enumerate(scales):
        cloud, tree = clouds[s]
        kk = min(k, len(cloud))
        for a in range(0, n, step):
            d, i = tree.query(
                query[a : a + step], k=kk, distance_upper_bound=s, workers=1
            )
            if d.ndim == 1:
                d, i = d[:, None], i[:, None]
            m = np.isfinite(d)
            i = np.where(m, i, 0)
            nb = cloud[i]
            c = m.sum(1)
            w = m[..., None]
            mu = (nb * w).sum(1) / np.maximum(c, 1)[:, None]
            dc = (nb - mu[:, None]) * w
            cov = (
                np.einsum("nki,nkj->nij", dc, dc)
                / np.maximum(c, 1)[:, None, None]
                + np.eye(3) * 1e-12
            )
            e = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0, None))[:, ::-1]
            e1 = np.maximum(e[:, 0], 1e-9)
            a1 = (e[:, 0] - e[:, 1]) / e1
            a2 = (e[:, 1] - e[:, 2]) / e1
            # Fewer than 3 neighbours: no shape, treat as a blob.
            few = c < 3
            a1[few], a2[few] = 0.0, 0.0
            feat[a : a + step, 2 * j] = a1
            feat[a : a + step, 2 * j + 1] = a2
            if s == count_scale:
                cnt1[a : a + step] = c
    return feat, cnt1


def lda_fit(xg, xv, ridge=1e-4):
    """Fit a linear classifier separating ground (`xg`) from vegetation
    (`xv`) samples.

    1. Fisher LDA: find the weights w that best separate the two classes,
       with ground scoring high.
    2. Choose the cut-off t on x @ w with the best balanced accuracy.

    Returns:
        w: weights.
        t: cut-off; x @ w >= t means ground.
        bacc: balanced accuracy on the samples (0.5 to 1).
    """
    mg, mv = xg.mean(0), xv.mean(0)
    sw = np.cov(xg, rowvar=False) + np.cov(xv, rowvar=False)
    sw += ridge * np.trace(sw) / len(sw) * np.eye(len(sw))
    w = np.linalg.solve(sw, mg - mv)
    pg, pv = xg @ w, xv @ w
    qs = np.linspace(0.01, 0.99, 197)
    cand = np.unique(np.quantile(np.concatenate([pg, pv]), qs))
    bacc = [(np.mean(pg >= t) + np.mean(pv < t)) / 2 for t in cand]
    t = cand[int(np.argmax(bacc))]
    return w, t, float(max(bacc))
