"""The ground filters.

Most people only need `ground_mask(xyz)` at the bottom of this file. It takes raw points and returns True for
every point it thinks is ground. Nothing here is trained on labelled data; all settings are fixed numbers
in `Settings`.

There are three filters and one way of combining them:

- shape_rule: a point is ground if its surroundings are flat, face the same way as the terrain, and sit
  low. It is cautious: stems and shrubs are rarely flat, so it seldom removes trees, but it does nibble at
  stem bases.
- ss_canupo ("self-seeded CANUPO"): every chunk trains its own small classifier. It takes the lowest points
  as examples of ground and points high up as examples of vegetation, then judges everything else by the
  shape of its neighbourhood at many scales. It finds more ground and spares more stem bases than the
  shape rule. It goes wrong when the lowest points are not ground, e.g. scans whose ground was already
  removed, because it then learns stem bases as "ground".
- coverage: tells the two situations apart. On real ground the shape rule finds something in nearly every
  2 m cell; without ground it only finds scattered bits.
- routed (the default): use ss_canupo when coverage says there is ground, and the shape rule otherwise.

All filters below work on a cloud that is already thinned to 0.05 m voxels and centred; `ground_mask` takes
care of that for you.
"""
from dataclasses import dataclass

import numpy as np

from .features import cell_index, ground_surface, lda_fit, multiscale, planarity, scale_clouds, voxel_first


@dataclass(frozen=True)
class Settings:
    """All the numbers the filters use. The defaults are what the benchmark results were produced with."""

    grid: float = 0.05            # points are thinned to one per 5 cm voxel
    cell: float = 2.0             # the "local floor" is the lowest point in each 2 m column
    hmax: float = 1.5             # nothing higher than 1.5 m above that floor can be ground
    fallback_h: float = 0.5       # when a point is too isolated to judge, call it ground if below 0.5 m
    min_nb: int = 5               # "too isolated" means fewer than 5 neighbours

    # shape rule
    rule_scale: float = 1.2       # look at neighbours within 1.2 m
    rule_planarity: float = 0.3   # how flat that neighbourhood must be (0 = not at all, 1 = perfect sheet)
    rule_angle: float = 20.0      # and it may tilt at most 20 degrees away from the terrain

    # self-seeded CANUPO
    scales: tuple = (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)  # radii of the shape fingerprint (m)
    seed_ground_h: float = 0.10   # points this close to the floor are the ground examples
    seed_veg_h: float = 3.0       # points this high above it are the vegetation examples
    n_seed: int = 20_000          # how many examples of each to train on
    min_seed: int = 500           # with fewer examples than this, skip the classifier and use the height rule
    count_scale: float = 1.0      # radius at which we check whether a point is too isolated
    seed: int = 0                 # makes the random choice of examples repeatable

    # routing
    coverage_cut: float = 0.63    # chunks above this coverage count as "has ground". On FOR-instance, chunks
                                  # without ground never went above 0.49 and chunks with ground never fell
                                  # below 0.77, so 0.63 sits in the middle of that gap.


DEFAULT = Settings()


def shape_rule(v, s=DEFAULT):
    """Ground = low, flat and facing the same way as the terrain."""
    h, tnorm = ground_surface(v, s.cell)
    g = np.zeros(len(v), bool)
    cand = np.flatnonzero(h < s.hmax)
    if len(cand):
        f, _ = voxel_first(v, s.rule_scale / 6)
        plan, nrm, cnt = planarity(v[f], v[cand], s.rule_scale)
        cosang = np.abs((nrm * tnorm[cand]).sum(1))
        gc = (plan >= s.rule_planarity) & (cosang >= np.cos(np.deg2rad(s.rule_angle))) & (h[cand] < s.hmax)
        g[cand] = np.where(cnt < s.min_nb, h[cand] < s.fallback_h, gc)
    return g


def ss_canupo(v, s=DEFAULT, return_info=False):
    """Self-seeded CANUPO: train a classifier on this chunk's own lowest and highest points, then apply it.

    Only safe when the lowest points really are ground; `routed` checks that first.
    """
    h, _ = ground_surface(v, s.cell)
    clouds = scale_clouds(v, s.scales, s.grid)
    fallback = h < s.fallback_h
    rng = np.random.default_rng(s.seed)
    gi, vi = np.flatnonzero(h < s.seed_ground_h), np.flatnonzero(h > s.seed_veg_h)
    info = dict(n_seed_ground=len(gi), n_seed_veg=len(vi), classifier=False)
    if len(gi) < s.min_seed or len(vi) < s.min_seed:
        return (fallback, info) if return_info else fallback
    gi = rng.choice(gi, min(s.n_seed, len(gi)), replace=False)
    vi = rng.choice(vi, min(s.n_seed, len(vi)), replace=False)
    xg, _ = multiscale(clouds, s.scales, v[gi], count_scale=s.count_scale)
    xv, _ = multiscale(clouds, s.scales, v[vi], count_scale=s.count_scale)
    w, t, bacc = lda_fit(xg.astype(np.float64), xv.astype(np.float64))
    cand = np.flatnonzero(h < s.hmax)
    x, c1 = multiscale(clouds, s.scales, v[cand], count_scale=s.count_scale)
    pred = np.zeros(len(v), bool)
    pred[cand] = np.where(c1 < s.min_nb, fallback[cand], (x.astype(np.float64) @ w) >= t)
    info.update(classifier=True, seed_balanced_accuracy=round(bacc, 4))
    return (pred, info) if return_info else pred


def coverage(v, mask, s=DEFAULT):
    """Of all 2 m columns that contain points, the share in which `mask` marks at least one point."""
    occ, cinv = np.unique(cell_index(v[:, :2], s.cell), return_inverse=True)
    return len(np.unique(cinv[mask])) / len(occ)


def routed(v, s=DEFAULT, return_info=False):
    """Run the shape rule, check whether the chunk has ground, and if so hand over to self-seeded CANUPO."""
    rule = shape_rule(v, s)
    cov = coverage(v, rule, s)
    info = dict(coverage=round(cov, 4), method="shape_rule")
    if cov >= s.coverage_cut:
        g, ss_info = ss_canupo(v, s, return_info=True)
        info.update(ss_info, method="ss_canupo")
    else:
        g = rule
    return (g, info) if return_info else g


METHODS = {"routed": routed, "ss_canupo": ss_canupo, "shape_rule": shape_rule,
           "height": lambda v, s=DEFAULT: ground_surface(v, s.cell)[0] < s.fallback_h}


def ground_mask(xyz, method="routed", settings=DEFAULT, return_info=False):
    """Find the ground points in a point cloud.

    Give it an (n, 3) array of coordinates in metres (any offset, any density) and you get back n booleans,
    True for ground. Internally the cloud is thinned to 5 cm voxels, filtered, and every original point
    gets the answer of its voxel.

    `method` picks the filter: "routed" (default and recommended), or one of "ss_canupo", "shape_rule" and
    "height" (everything below 0.5 m) if you want to compare them. With `return_info=True` you also get a
    small dict saying which filter was used and the chunk's coverage.
    """
    xyz = np.asarray(xyz, np.float64)
    first, inv = voxel_first(xyz, settings.grid)
    v = xyz[first]
    v = v - v.mean(0)
    fn = METHODS[method]
    if return_info and method in ("routed", "ss_canupo"):
        g, info = fn(v, settings, return_info=True)
    else:
        g, info = fn(v, settings), {"method": method}
    info.update(n_points=len(xyz), n_voxels=len(v))
    return (g[inv], info) if return_info else g[inv]
