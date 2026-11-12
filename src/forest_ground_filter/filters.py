"""Ground filters.

Entry point: `ground_mask(xyz)` returns True for every ground point. No
labelled data is used; all settings are fixed values in `Settings`.

Filters:

- shape_rule: ground = low, flat and aligned with the terrain. Rarely
  removes trees; cuts some stem bases.
- ss_canupo: self-seeded, CANUPO-style. Trains a classifier per chunk on its
  lowest points (ground) and its high points (vegetation), then classifies
  the rest by multi-scale shape. Cuts fewer stem bases; fails if the lowest
  points are not ground (e.g. scans with the ground already removed).
- coverage: share of 2 m cells in which the shape rule finds ground; high
  when the chunk has ground, low when it has none.
- routed (default): ss_canupo if coverage >= `coverage_cut`, else
  shape_rule.

All filters take a cloud already thinned to 0.05 m voxels and centred;
`ground_mask` does this.
"""

from dataclasses import dataclass

import numpy as np

from .features import (
    cell_index,
    ground_surface,
    lda_fit,
    multiscale,
    planarity,
    scale_clouds,
    voxel_first,
)


@dataclass(frozen=True)
class Settings:
    """Filter settings. The defaults produce the benchmark results."""

    # Voxel size (m).
    grid: float = 0.05
    # Floor = lowest point of each column of this width (m).
    cell: float = 2.0
    # Maximum height of ground above the floor (m).
    hmax: float = 1.5
    # Isolated points: ground if below this height (m).
    fallback_h: float = 0.5
    # Isolated = fewer neighbours than this.
    min_nb: int = 5

    # Shape rule: radius (m), minimum planarity (0 to 1), and maximum angle
    # between surface normal and terrain normal (degrees).
    rule_scale: float = 1.2
    rule_planarity: float = 0.3
    rule_angle: float = 20.0

    # Self-seeded CANUPO.
    # Radii of the multi-scale descriptor (m).
    scales: tuple = (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    # Ground samples: below this height (m).
    seed_ground_h: float = 0.10
    # Vegetation samples: above this height (m).
    seed_veg_h: float = 3.0
    # Samples per class.
    n_seed: int = 20_000
    # Fewer samples than this: height rule for the whole chunk.
    min_seed: int = 500
    # Radius for the isolated-point check (m).
    count_scale: float = 1.0
    # Random seed for sample selection.
    seed: int = 0

    # Routing: coverage at or above this means the chunk has ground.
    # FOR-instance tiles: <= 0.49 without ground, >= 0.77 with ground.
    coverage_cut: float = 0.63


DEFAULT = Settings()


def shape_rule(v, s=DEFAULT):
    """Shape rule: ground = low, flat and aligned with the terrain.

    1. Candidates: points below `hmax`.
    2. Planarity and normal at radius `rule_scale`.
    3. Ground if planarity >= `rule_planarity` and the normal is within
       `rule_angle` of the terrain normal.
    4. Isolated points: ground if below `fallback_h`.
    """
    h, tnorm = ground_surface(v, s.cell)
    g = np.zeros(len(v), bool)
    cand = np.flatnonzero(h < s.hmax)
    if len(cand):
        f, _ = voxel_first(v, s.rule_scale / 6)
        plan, nrm, cnt = planarity(v[f], v[cand], s.rule_scale)
        cosang = np.abs((nrm * tnorm[cand]).sum(1))
        flat = plan >= s.rule_planarity
        aligned = cosang >= np.cos(np.deg2rad(s.rule_angle))
        gc = flat & aligned & (h[cand] < s.hmax)
        g[cand] = np.where(cnt < s.min_nb, h[cand] < s.fallback_h, gc)
    return g


def ss_canupo(v, s=DEFAULT, return_info=False):
    """Self-seeded CANUPO.

    1. Samples: ground = below `seed_ground_h`, vegetation = above
       `seed_veg_h`; up to `n_seed` each.
    2. Multi-scale descriptor of the samples.
    3. Fit an LDA on the samples.
    4. Classify all points below `hmax`; isolated points use the height
       rule.

    Too few samples of either class: height rule for the whole chunk.
    Assumes the lowest points are ground; `routed` checks this first.
    """
    h, _ = ground_surface(v, s.cell)
    clouds = scale_clouds(v, s.scales, s.grid)
    fallback = h < s.fallback_h
    rng = np.random.default_rng(s.seed)
    gi = np.flatnonzero(h < s.seed_ground_h)
    vi = np.flatnonzero(h > s.seed_veg_h)
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
    score = x.astype(np.float64) @ w
    pred = np.zeros(len(v), bool)
    pred[cand] = np.where(c1 < s.min_nb, fallback[cand], score >= t)
    info.update(classifier=True, seed_balanced_accuracy=round(bacc, 4))
    return (pred, info) if return_info else pred


def coverage(v, mask, s=DEFAULT):
    """Share of occupied 2 m columns with at least one `mask` point."""
    cells = cell_index(v[:, :2], s.cell)
    occ, cinv = np.unique(cells, return_inverse=True)
    return len(np.unique(cinv[mask])) / len(occ)


def routed(v, s=DEFAULT, return_info=False):
    """Routed filter (default).

    1. Run the shape rule.
    2. Compute its coverage.
    3. Coverage >= `coverage_cut`: return self-seeded CANUPO.
       Otherwise: return the shape rule.
    """
    rule = shape_rule(v, s)
    cov = coverage(v, rule, s)
    info = dict(coverage=round(cov, 4), method="shape_rule")
    if cov >= s.coverage_cut:
        g, ss_info = ss_canupo(v, s, return_info=True)
        info.update(ss_info, method="ss_canupo")
    else:
        g = rule
    return (g, info) if return_info else g


def height(v, s=DEFAULT):
    """Height rule: ground = below `fallback_h` above the floor."""
    return ground_surface(v, s.cell)[0] < s.fallback_h


METHODS = {
    "routed": routed,
    "ss_canupo": ss_canupo,
    "shape_rule": shape_rule,
    "height": height,
}


def ground_mask(xyz, method="routed", settings=DEFAULT, return_info=False):
    """Return a ground mask for a point cloud.

    1. Thin `xyz` to `settings.grid` voxels and centre it.
    2. Run the filter `method`.
    3. Copy each voxel's label back to its points.

    Args:
        xyz: (n, 3) coordinates in metres.
        method: "routed" (default), "ss_canupo", "shape_rule" or "height".
        settings: `Settings`.
        return_info: also return a dict with the filter used and coverage.

    Returns:
        (n,) bool, True = ground.
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
