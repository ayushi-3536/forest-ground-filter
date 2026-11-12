"""Benchmark on TLS datasets: Cherlet, ForestSemantic and TreeScanPL10K.

Labels are used for scoring only.

- cherlet: Cherlet TLS instance benchmark (Zenodo 16875688), PLY fields
  `semantic` (0 ground, 1 tree) and `instance` (tree ID). Ground here
  probably includes understorey.
- forestsemantic: ForestSemantic Plot_1 (Zenodo 13285640), LAS
  `classification` (1 ground, 2 trunk, 3-4 branches, 5 foliage, 6 misc)
  and `point_source_id` (tree ID). Trees = 2-5; stems = 2.
- pl10k: TreeScanPL10K 5 cm plots (Zenodo 19127709), PLY field `treeID`.
  No ground class, so ground is a proxy: non-tree points less than 0.15 m
  above the floor. Non-tree points from 0.15 to 2 m are ignored; non-tree
  points above 2 m and all tree points are non-ground.

Each file is thinned to 0.05 m voxels, cut into 20 m tiles, and every tile
is filtered on its own by every method in `methods.py`. Per file this
writes one JSON; `--summarize` pools them into summary.md.

    python benchmarks/tls_sets.py --dataset cherlet --file <ply> --out <dir>
    python benchmarks/tls_sets.py --summarize --out <dir>
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from forest_ground_filter import DEFAULT
from forest_ground_filter.features import ground_surface, voxel_first
from methods import METHODS, filter_tile

TILE = 20.0
MIN_VOX = 10_000
PROXY_H = 0.15


def load(dataset, path):
    """Return a dict with xyz, ground, tree, stem, ignore and tree IDs."""
    if dataset == "cherlet":
        from plyfile import PlyData

        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        sem = np.asarray(v["semantic"]).astype(np.int64)
        tree = sem == 1
        tid = np.where(tree, np.asarray(v["instance"]).astype(np.int64), 0)
        # No stem class: stem bases = tree points low above the floor.
        return dict(
            xyz=xyz,
            ground=sem == 0,
            tree=tree,
            stem=tree,
            ignore=np.zeros(len(sem), bool),
            tid=tid,
        )
    if dataset == "forestsemantic":
        import laspy

        las = laspy.read(path)
        print("LAS dimensions:", list(las.point_format.dimension_names))
        xyz = np.stack([las.x, las.y, las.z], 1)
        c = np.asarray(las.classification).astype(np.int64)
        tree = np.isin(c, (2, 3, 4, 5))
        tid = np.where(tree, np.asarray(las.point_source_id), 0)
        return dict(
            xyz=xyz,
            ground=c == 1,
            tree=tree,
            stem=c == 2,
            ignore=~np.isin(c, (1, 2, 3, 4, 5, 6)),
            tid=tid.astype(np.int64),
        )
    if dataset == "pl10k":
        from plyfile import PlyData

        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        tid = np.asarray(v["treeID"]).astype(np.int64)
        tree = tid > 0
        # Ground and ignore are set after voxelisation (they need heights).
        return dict(
            xyz=xyz, ground=None, tree=tree, stem=tree, ignore=None, tid=tid
        )
    raise ValueError(dataset)


def counts(pred, xyz, d):
    keep = ~d["ignore"]
    ground, tree = d["ground"], d["tree"]
    h, _ = ground_surface(xyz)
    base = d["stem"] & (h < 2.0)
    under = ~ground & ~tree & keep
    r = dict(
        tp=int((pred & ground & keep).sum()),
        fp=int((pred & ~ground & keep).sum()),
        fn=int((~pred & ground & keep).sum()),
        tree_pts=int(tree.sum()),
        tree_removed=int((pred & tree).sum()),
        base_pts=int(base.sum()),
        base_removed=int((pred & base).sum()),
        under_pts=int(under.sum()),
        under_removed=int((pred & under).sum()),
        trees=0,
        trees_killed=0,
    )
    tid = d["tid"]
    if tid.any():
        ut, tinv = np.unique(tid, return_inverse=True)
        size = np.bincount(tinv, minlength=len(ut))
        rem = np.bincount(tinv[pred], minlength=len(ut))
        big = (ut > 0) & (size >= 50)
        r["trees"] = int(big.sum())
        r["trees_killed"] = int((big & (rem >= 0.5 * size)).sum())
    return r


def run_file(dataset, path, out, workers):
    t0 = time.time()
    name = os.path.splitext(os.path.basename(path))[0]
    d = load(dataset, path)
    first, _ = voxel_first(d["xyz"], DEFAULT.grid)
    d = {k: (v[first] if v is not None else None) for k, v in d.items()}
    xyz = d.pop("xyz")
    xyz = xyz - xyz.mean(0)
    if dataset == "pl10k":
        h, _ = ground_surface(xyz)
        d["ground"] = ~d["tree"] & (h < PROXY_H)
        d["ignore"] = ~d["tree"] & (h >= PROXY_H) & (h <= 2.0)
    print(f"{name}: {len(xyz)} voxels", flush=True)

    tij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / TILE).astype(np.int64)
    key = tij[:, 0] * 10_000 + tij[:, 1]
    tiles = [np.flatnonzero(key == k) for k in np.unique(key)]
    tiles = [t for t in tiles if len(t) >= MIN_VOX]
    preds = {m: np.zeros(len(xyz), bool) for m in METHODS}
    with Pool(workers) as pool:
        results = pool.map(filter_tile, [xyz[t] for t in tiles])
    covs, secs = [], {m: 0.0 for m in METHODS}
    for t, (masks, cov, sec) in zip(tiles, results):
        covs.append(cov)
        for m in METHODS:
            preds[m][t] = masks[m]
            secs[m] += sec[m]
    rows = {m: counts(p, xyz, d) for m, p in preds.items()}
    res = dict(
        dataset=dataset,
        file=name,
        voxels=len(xyz),
        tiles=len(tiles),
        coverages=covs,
        rows=rows,
        sec=secs,
        wall=round(time.time() - t0, 1),
    )
    with open(os.path.join(out, f"{dataset}__{name}.json"), "w") as fh:
        json.dump(res, fh)
    print(f"{name}: {len(tiles)} tiles, {res['wall']} s", flush=True)


def pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "–"


def table(rs, methods):
    lines = [
        "| Filter | P | R | F1 | tree pts removed | stem bases removed "
        "| understorey removed | trees killed | s / tile |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ntiles = max(sum(r["tiles"] for r in rs), 1)
    for m in methods:
        t = {k: sum(r["rows"][m][k] for r in rs) for k in rs[0]["rows"][m]}
        P = t["tp"] / max(t["tp"] + t["fp"], 1)
        R = t["tp"] / max(t["tp"] + t["fn"], 1)
        F = 2 * P * R / max(P + R, 1e-9)
        killed = f"{t['trees_killed']}/{t['trees']}" if t["trees"] else "–"
        sec = sum(r["sec"][m] for r in rs) / ntiles
        lines.append(
            f"| {m} | {100 * P:.1f} | {100 * R:.1f} | {100 * F:.1f} "
            f"| {pct(t['tree_removed'], t['tree_pts'])} "
            f"| {pct(t['base_removed'], t['base_pts'])} "
            f"| {pct(t['under_removed'], t['under_pts'])} | {killed} "
            f"| {sec:.1f} |"
        )
    return lines


def summarize(out):
    pattern = os.path.join(out, "*__*.json")
    res = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    lines = [
        "# Ground filters on TLS datasets",
        "",
        "pl10k: proxy ground (non-tree points < 0.15 m above the "
        "floor), not a ground label.",
        "",
    ]
    for ds in sorted({r["dataset"] for r in res}):
        rs = [r for r in res if r["dataset"] == ds]
        methods = list(rs[0]["rows"])
        cov = np.concatenate([r["coverages"] for r in rs if r["coverages"]])
        lines += [
            f"## {ds}: {len(rs)} files, "
            f"{sum(r['tiles'] for r in rs)} tiles",
            "",
        ]
        lines += table(rs, methods)
        lines += [
            "",
            f"Coverage per tile: min {cov.min():.2f}, "
            f"5th pct {np.percentile(cov, 5):.2f}, "
            f"median {np.median(cov):.2f}; "
            f"tiles below the cut-off: "
            f"{int((cov < DEFAULT.coverage_cut).sum())}",
            "",
        ]
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", choices=["cherlet", "forestsemantic", "pl10k"]
    )
    ap.add_argument("--file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
    else:
        print("methods:", METHODS, flush=True)
        run_file(a.dataset, a.file, a.out, a.workers)


if __name__ == "__main__":
    main()
