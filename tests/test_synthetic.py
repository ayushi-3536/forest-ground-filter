"""Synthetic 20 m forest chunk: sloped, bumpy floor + stems + crowns + low shrubs."""
import numpy as np

from forest_ground_filter import ground_mask


def scene(with_ground=True, seed=0):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    if with_ground:
        xy = rng.uniform(0, 20, (160_000, 2))
        z = 0.15 * xy[:, 0] + 0.05 * np.sin(xy[:, 1]) + rng.normal(0, 0.01, len(xy))  # ~8.5 deg slope
        parts.append(np.c_[xy, z]); labels.append(np.zeros(len(xy), int))
    ground_z = lambda x, y: 0.15 * x + 0.05 * np.sin(y)  # noqa: E731
    for cx, cy in rng.uniform(2, 18, (15, 2)):
        zb = ground_z(cx, cy)
        a = rng.uniform(0, 2 * np.pi, 4000); hh = rng.uniform(0, 12, 4000)
        parts.append(np.c_[cx + 0.15 * np.cos(a), cy + 0.15 * np.sin(a), zb + hh]); labels.append(np.ones(4000, int))
        c = rng.normal(0, 1.5, (3000, 3)) + [cx, cy, zb + 9]
        parts.append(c); labels.append(np.full(3000, 2))
    for cx, cy in rng.uniform(1, 19, (20, 2)):
        b = rng.normal(0, 0.25, (600, 3)) * [1, 1, 0.6] + [cx, cy, ground_z(cx, cy) + 0.35]
        parts.append(b); labels.append(np.full(600, 3))
    return np.concatenate(parts), np.concatenate(labels)


def test_with_ground():
    xyz, lab = scene(True)
    g, info = ground_mask(xyz, return_info=True)
    assert info["method"] == "ss_canupo", info
    assert g[lab == 0].mean() > 0.9, g[lab == 0].mean()          # ground found
    stem_base = (lab == 1) & (xyz[:, 2] - (0.15 * xyz[:, 0] + 0.05 * np.sin(xyz[:, 1])) < 2.0)
    assert g[stem_base].mean() < 0.10, g[stem_base].mean()      # stem bases kept
    assert g[lab == 2].mean() < 0.001                            # crowns kept


def test_without_ground():
    xyz, lab = scene(False)
    g, info = ground_mask(xyz, return_info=True)
    assert info["method"] == "shape_rule", info                 # coverage gate detects the missing floor
    assert g.mean() < 0.02, g.mean()                            # almost nothing removed


def test_methods_run():
    xyz, _ = scene(True, seed=1)
    for m in ("routed", "ss_canupo", "shape_rule", "height"):
        g = ground_mask(xyz, method=m)
        assert g.shape == (len(xyz),) and g.dtype == bool
