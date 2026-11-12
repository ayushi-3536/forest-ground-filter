"""Benchmark on labelled TLS/MLS datasets with a ground class.

Datasets (labels used for scoring only):

- segmentedforests: SegmentedForests (Zenodo 17396681), LAZ field `Class`.
  Ground = class 1 (ground and ground vegetation). Trees = branches/leaves,
  stems, standing dead trees, small trees (2, 3, 10, 12, 13, 14, 22, 23);
  stems = 3, 10, 13, 14, 23. Ignored: poles, people, other, boulders
  (6, 7, 8, 9). No tree IDs, so no "trees killed".
- cherlet: Cherlet TLS instance benchmark (Zenodo 16875688), PLY fields
  `semantic` (0 ground, 1 tree) and `instance` (tree ID).
- forestsemantic: ForestSemantic Plot_1 (Zenodo 13285640), LAS
  `classification` (1 ground, 2 trunk, 3-4 branches, 5 foliage, 6 misc)
  and `point_source_id` (tree ID). Trees = 2-5; stems = 2.

Each file is thinned to 0.05 m voxels, cut into 20 m tiles, and every tile
is filtered on its own. Per file this writes one JSON with the counts per
filter; `--summarize` pools them into summary.md.

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

from forest_ground_filter import DEFAULT, coverage, shape_rule, ss_canupo
from forest_ground_filter.features import ground_surface, voxel_first
from forest_ground_filter.filters import height

TILE = 20.0
MIN_VOX = 10_000
METHODS = ("routed", "ss_canupo", "shape_rule", "height")
SF_TREE = (2, 3, 10, 12, 13, 14, 22, 23)
SF_STEM = (3, 10, 13, 14, 23)
SF_IGNORE = (6, 7, 8, 9)


def load(dataset, path):
    """Return xyz, ground, tree, stem, ignore (bool) and tree ID (int)."""
    if dataset == "segmentedforests":
        import laspy

        las = laspy.read(path)
        xyz = np.stack([las.x, las.y, las.z], 1)
        c = np.asarray(las["Class"]).astype(np.int64)
        tid = np.zeros(len(c), np.int64)
        return (
            xyz,
            c == 1,
            np.isin(c, SF_TREE),
            np.isin(c, SF_STEM),
            np.isin(c, SF_IGNORE),
            tid,
        )
    if dataset == "cherlet":
        from plyfile import PlyData

        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        sem = np.asarray(v["semantic"]).astype(np.int64)
        tid = np.asarray(v["instance"]).astype(np.int64)
        tree = sem == 1
        tid = np.where(tree, tid, 0)
        # No stem class: all tree points count; stem bases come from height.
        return xyz, sem == 0, tree, tree, np.zeros(len(sem), bool), tid
    if dataset == "forestsemantic":
        import laspy

        las = laspy.read(path)
        print("LAS dimensions:", list(las.point_format.dimension_names))
        xyz = np.stack([las.x, las.y, las.z], 1)
        c = np.asarray(las.classification).astype(np.int64)
        tid = np.asarray(las.point_source_id).astype(np.int64)
        tree = np.isin(c, (2, 3, 4, 5))
        tid = np.where(tree, tid, 0)
        return (
            xyz,
            c == 1,
            tree,
            c == 2,
            ~np.isin(c, (1, 2, 3, 4, 5, 6)),
            tid,
        )
    raise ValueError(dataset)


def filter_tile(xyz):
    """All filters on one tile; `routed` is derived as in `ground_mask`."""
    s = DEFAULT
    first, inv = voxel_first(xyz, s.grid)
    v = xyz[first]
    v = v - v.mean(0)
    rule = shape_rule(v, s)
    cov = coverage(v, rule, s)
    ss = ss_canupo(v, s)
    out = {
        "shape_rule": rule,
        "ss_canupo": ss,
        "height": height(v, s),
        "routed": ss if cov >= s.coverage_cut else rule,
    }
    return {m: g[inv] for m, g in out.items()}, cov


def counts(pred, xyz, ground, tree, stem, ignore, tid):
    keep = ~ignore
    h, _ = ground_surface(xyz)
    base = stem & (h < 2.0)
    understorey = ~ground & ~tree & keep
    r = dict(
        tp=int((pred & ground & keep).sum()),
        fp=int((pred & ~ground & keep).sum()),
        fn=int((~pred & ground & keep).sum()),
        tree_pts=int(tree.sum()),
        tree_removed=int((pred & tree).sum()),
        base_pts=int(base.sum()),
        base_removed=int((pred & base).sum()),
        under_pts=int(understorey.sum()),
        under_removed=int((pred & understorey).sum()),
        trees=0,
        trees_killed=0,
    )
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
    xyz, ground, tree, stem, ignore, tid = load(dataset, path)
    first, _ = voxel_first(xyz, DEFAULT.grid)
    sel = (
        xyz[first],
        ground[first],
        tree[first],
        stem[first],
        ignore[first],
        tid[first],
    )
    xyz, ground, tree, stem, ignore, tid = sel
    xyz = xyz - xyz.mean(0)
    print(f"{name}: {len(xyz)} voxels", flush=True)
    tij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / TILE).astype(np.int64)
    key = tij[:, 0] * 10_000 + tij[:, 1]
    tiles = [np.flatnonzero(key == k) for k in np.unique(key)]
    tiles = [t for t in tiles if len(t) >= MIN_VOX]
    preds = {m: np.zeros(len(xyz), bool) for m in METHODS}
    with Pool(workers) as pool:
        results = pool.map(filter_tile, [xyz[t] for t in tiles])
    covs = []
    for t, (masks, cov) in zip(tiles, results):
        covs.append(cov)
        for m in METHODS:
            preds[m][t] = masks[m]
    rows = {
        m: counts(p, xyz, ground, tree, stem, ignore, tid)
        for m, p in preds.items()
    }
    res = dict(
        dataset=dataset,
        file=name,
        voxels=len(xyz),
        tiles=len(tiles),
        coverages=covs,
        rows=rows,
        sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(out, f"{dataset}__{name}.json"), "w") as fh:
        json.dump(res, fh)
    print(f"{name}: {len(tiles)} tiles, {res['sec']} s", flush=True)


def pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "–"


def summarize(out):
    res = [
        json.load(open(f))
        for f in sorted(glob.glob(os.path.join(out, "*__*.json")))
    ]
    lines = [
        "# forest-ground-filter on labelled TLS/MLS datasets",
        "",
        "| Dataset (files, tiles) | Filter | P | R | F1 "
        "| tree pts removed | stem bases removed "
        "| understorey removed | trees killed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in sorted({r["dataset"] for r in res}):
        rs = [r for r in res if r["dataset"] == ds]
        ntiles = sum(r["tiles"] for r in rs)
        for m in METHODS:
            t = {k: sum(r["rows"][m][k] for r in rs) for k in rs[0]["rows"][m]}
            P = t["tp"] / max(t["tp"] + t["fp"], 1)
            R = t["tp"] / max(t["tp"] + t["fn"], 1)
            F = 2 * P * R / max(P + R, 1e-9)
            killed = f"{t['trees_killed']}/{t['trees']}" if t["trees"] else "–"
            lines.append(
                f"| {ds} ({len(rs)}, {ntiles}) | {m} | {100 * P:.1f} "
                f"| {100 * R:.1f} | {100 * F:.1f} "
                f"| {pct(t['tree_removed'], t['tree_pts'])} "
                f"| {pct(t['base_removed'], t['base_pts'])} "
                f"| {pct(t['under_removed'], t['under_pts'])} | {killed} |"
            )
    lines += [
        "",
        "Per file, ground F1 (routed / shape_rule / ss_canupo) "
        "and coverage (min / median):",
        "",
    ]
    for r in res:
        f1 = []
        for m in ("routed", "shape_rule", "ss_canupo"):
            c = r["rows"][m]
            f1.append(
                100 * 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)
            )
        cv = np.array(r["coverages"]) if r["coverages"] else np.zeros(1)
        lines.append(
            f"- {r['dataset']} {r['file']}: {f1[0]:.1f} / {f1[1]:.1f} / "
            f"{f1[2]:.1f}; coverage {cv.min():.2f} / {np.median(cv):.2f}"
        )
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", choices=["segmentedforests", "cherlet", "forestsemantic"]
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
        run_file(a.dataset, a.file, a.out, a.workers)


if __name__ == "__main__":
    main()
