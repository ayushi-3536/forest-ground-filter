"""Ground filters for forest point clouds, and the routed combination used by default.

All filters take a VOXELISED, CENTRED cloud `v` (n x 3, one point per 0.05 m voxel; see `ground_mask` for raw
input) and return a bool "is ground" per point. Nothing is trained on labelled data: every setting is a fixed
constant of `Settings`.

shape_rule    planar (a2D >= 0.3 at r = 1.2 m), facing the local terrain (normal within 20 deg) and low (< 1.5 m
              above the 2 m-cell minimum); fewer than 5 neighbours -> height < 0.5 m. Safe on scenes without
              ground (it rarely calls stems or shrubs ground), but cuts some stem bases.
ss_canupo     self-seeded CANUPO: CANUPO's multi-scale dimensionality (Brodu & Lague 2012; a1D, a2D at 10 scales,
              0.1-2 m) with an LDA trained per chunk on height pseudo-labels (bottom 0.10 m = ground, > 3 m =
              vegetation). Better ground recall and fewer stem bases cut, but wrong where the bottom layer is not
              ground (ground-free scenes: it then learns stem bases as ground).
coverage      share of occupied 2 m cells in which the shape rule finds ground: high on real ground (>= 0.77 on
              all 351 FOR-instance ground tiles), low without (<= 0.49 on all 170 no-ground tiles).
routed        ss_canupo if coverage >= 0.63, else shape_rule.  <- default
"""
from dataclasses import dataclass

import numpy as np

from .features import cell_index, ground_surface, lda_fit, multiscale, planarity, scale_clouds, voxel_first


@dataclass(frozen=True)
class Settings:
    grid: float = 0.05            # voxel size (m)
    cell: float = 2.0             # floor cells (m)
    hmax: float = 1.5             # ground only below this height above the cell minimum (m)
    fallback_h: float = 0.5       # height rule for points too sparse to measure shape (m)
    min_nb: int = 5               # fewer neighbours than this -> fallback
    # shape rule
    rule_scale: float = 1.2       # PCA radius (m)
    rule_planarity: float = 0.3   # a2D threshold
    rule_angle: float = 20.0      # max angle between the surface normal and the terrain normal (deg)
    # SS-CANUPO
    scales: tuple = (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    seed_ground_h: float = 0.10   # ground pseudo-labels: below this height (m)
    seed_veg_h: float = 3.0       # vegetation pseudo-labels: above this height (m)
    n_seed: int = 20_000          # pseudo-labels per class
    min_seed: int = 500           # fewer -> height fallback for the whole chunk
    count_scale: float = 1.0      # scale whose neighbour count decides the sparse fallback
    seed: int = 0                 # random seed for the pseudo-label sample
    # routing
    coverage_cut: float = 0.63    # middle of the FOR-instance gap (0.49 no-ground max, 0.77 ground min)


DEFAULT = Settings()


def shape_rule(v, s=DEFAULT):
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
    """Share of occupied `s.cell` columns that contain at least one point of `mask`."""
    occ, cinv = np.unique(cell_index(v[:, :2], s.cell), return_inverse=True)
    return len(np.unique(cinv[mask])) / len(occ)


def routed(v, s=DEFAULT, return_info=False):
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
    """Bool "is ground" per input point. `xyz`: (n, 3) raw coordinates, any offset or density.

    The cloud is voxelised at `settings.grid` (first point per voxel), filtered, and every input point gets
    its voxel's label. `method`: "routed" (default), "ss_canupo", "shape_rule" or "height"."""
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
