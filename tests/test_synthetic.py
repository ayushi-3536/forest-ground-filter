"""Tests on a synthetic 20 m chunk: sloped floor, stems, crowns, shrubs."""

import numpy as np

from forest_ground_filter import ground_mask


def ground_z(x, y):
    """Floor height: ~8.5 degree slope plus bumps."""
    return 0.15 * x + 0.05 * np.sin(y)


def scene(with_ground=True, seed=0):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    if with_ground:
        xy = rng.uniform(0, 20, (160_000, 2))
        z = ground_z(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.01, len(xy))
        parts.append(np.c_[xy, z])
        labels.append(np.zeros(len(xy), int))
    # 15 trees: a 12 m stem of radius 0.15 m and a crown around 9 m.
    for cx, cy in rng.uniform(2, 18, (15, 2)):
        zb = ground_z(cx, cy)
        a = rng.uniform(0, 2 * np.pi, 4000)
        hh = rng.uniform(0, 12, 4000)
        stem = np.c_[cx + 0.15 * np.cos(a), cy + 0.15 * np.sin(a), zb + hh]
        parts.append(stem)
        labels.append(np.ones(4000, int))
        parts.append(rng.normal(0, 1.5, (3000, 3)) + [cx, cy, zb + 9])
        labels.append(np.full(3000, 2))
    # 20 low shrubs.
    for cx, cy in rng.uniform(1, 19, (20, 2)):
        centre = [cx, cy, ground_z(cx, cy) + 0.35]
        parts.append(rng.normal(0, 0.25, (600, 3)) * [1, 1, 0.6] + centre)
        labels.append(np.full(600, 3))
    return np.concatenate(parts), np.concatenate(labels)


def test_with_ground():
    xyz, lab = scene(True)
    g, info = ground_mask(xyz, return_info=True)
    assert info["method"] == "ss_canupo", info
    # Floor found.
    assert g[lab == 0].mean() > 0.9, g[lab == 0].mean()
    # Stem bases and crowns kept.
    height = xyz[:, 2] - ground_z(xyz[:, 0], xyz[:, 1])
    stem_base = (lab == 1) & (height < 2.0)
    assert g[stem_base].mean() < 0.10, g[stem_base].mean()
    assert g[lab == 2].mean() < 0.001


def test_without_ground():
    xyz, _ = scene(False)
    g, info = ground_mask(xyz, return_info=True)
    # Coverage detects the missing floor.
    assert info["method"] == "shape_rule", info
    # Almost nothing removed.
    assert g.mean() < 0.02, g.mean()


def test_methods_run():
    xyz, _ = scene(True, seed=1)
    for m in ("routed", "ss_canupo", "shape_rule", "height"):
        g = ground_mask(xyz, method=m)
        assert g.shape == (len(xyz),) and g.dtype == bool
