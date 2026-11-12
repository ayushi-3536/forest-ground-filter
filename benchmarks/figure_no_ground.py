"""Figure: what each filter removes on a chunk without ground.

Takes the FOR-instance BlueCat test plot (TLS, no ground class), picks the
20 m chunk with the most trees, runs every filter in `methods.py`, and draws
a 3 m thick side slice. Removed points are red; since the plot has no ground,
every red point is a tree point.

    python benchmarks/figure_no_ground.py --ply <BlueCat_..._test.ply> \
        --out docs/no_ground_example.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plyfile import PlyData  # noqa: E402

from forest_ground_filter import DEFAULT  # noqa: E402
from forest_ground_filter.features import (  # noqa: E402
    ground_surface,
    voxel_first,
)
from methods import filter_tile  # noqa: E402

PANELS = [
    ("routed", "SS-CANUPO (routed, default)"),
    ("shape_rule", "Shape rule"),
    ("ss_canupo", "SS-CANUPO without ground check"),
    ("height", "Height < 0.5 m"),
    ("csf_default", "CSF"),
    ("csf_relief", "CSF (relief)"),
    ("smrf_default", "SMRF"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = PlyData.read(a.ply)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    tid = np.asarray(v["treeID"]).astype(np.int64)
    first, _ = voxel_first(xyz, DEFAULT.grid)
    xyz, tid = xyz[first] - xyz[first].mean(0), tid[first]

    # The 20 m chunk with the most trees.
    tij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / 20.0).astype(np.int64)
    key = tij[:, 0] * 10_000 + tij[:, 1]

    def n_trees(k):
        return len(np.unique(tid[(key == k) & (tid > 0)]))

    idx = np.flatnonzero(key == max(np.unique(key), key=n_trees))
    t, ttid = xyz[idx], tid[idx]
    masks, _, _ = filter_tile(t)
    h, _ = ground_surface(t)
    base = (ttid > 0) & (h < 2.0)

    # 3 m thick slice through the chunk centre, lowest 8 m.
    sl = (np.abs(t[:, 1] - np.median(t[:, 1])) < 1.5) & (h < 8.0)
    kw = dict(linewidths=0, rasterized=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharex=True, sharey=True)
    axes.flat[0].scatter(t[sl, 0], h[sl], s=0.4, c="#4a7c59", **kw)
    axes.flat[0].set_title("Input (no ground)")
    for ax, (m, name) in zip(axes.flat[1:], PANELS):
        g = masks[m]
        pct = 100 * (g & base).sum() / max(base.sum(), 1)
        ax.scatter(t[sl & ~g, 0], h[sl & ~g], s=0.4, c="#bdbdbd", **kw)
        ax.scatter(t[sl & g, 0], h[sl & g], s=1.2, c="#d62728", **kw)
        ax.set_title(f"{name}\nstem bases removed: {pct:.1f} %")
    for ax in axes.flat:
        ax.set_ylim(-0.3, 8)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("height (m)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
