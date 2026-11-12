"""Benchmark on FOR-instance.

Expects the FOR-instanceV2 layout with BlueCat:
<root>/{train_val_data,test_data}/*_{val,test}.ply, with vertex fields
x, y, z, semantic_seg (1 = ground) and treeID (> 0 = tree). The labels are
only used for scoring.

Two modes:

- tiles: every plot is cut into 20 m tiles and each tile is filtered on its
  own, as a corpus chunk would be. Scores are pooled over all test plots
  (29), the plots without BlueCat (28), and per sensor.
- exact: whole plots, shape rule and self-seeded CANUPO only. The per-plot
  counts are compared with reference JSON files from the original
  evaluation, to show that the package reproduces it exactly.

Scoring uses 0.05 m voxels. Stem bases are tree points less than 2 m above
the local floor; a tree (>= 50 voxels) counts as killed if it loses at least
half of its voxels.

    python benchmarks/forinstance.py --root <dir> --out <dir> --mode tiles
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from plyfile import PlyData

from forest_ground_filter import DEFAULT, shape_rule, ss_canupo
from forest_ground_filter.features import ground_surface, voxel_first
from methods import METHODS, filter_tile

TILE = 20.0
MIN_VOX = 10_000
KEYS = (
    "tp",
    "fp",
    "fn",
    "tree_pts",
    "tree_removed",
    "base_pts",
    "base_removed",
    "trees",
    "trees_killed",
)
HEADER = [
    "| Filter | P | R | F1 | tree pts removed | stem bases removed "
    "| trees killed |",
    "|---|---|---|---|---|---|---|",
]
# Row keys used by the original evaluation scripts.
REF_KEYS = {
    "shape_rule": "canupo_s1.2_p0.3_a20_h1.5_rel",
    "ss_canupo": "canupo_lda",
}


def sensor(name):
    if name.startswith("BlueCat"):
        return "TLS"
    if name.startswith("NIBIO_MLS"):
        return "MLS"
    return "ULS"


def load(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    sem = np.asarray(v["semantic_seg"]).astype(np.int64)
    tid = np.asarray(v["treeID"]).astype(np.int64)
    first, _ = voxel_first(xyz, DEFAULT.grid)
    xyz, sem, tid = xyz[first], sem[first], tid[first]
    return xyz - xyz.mean(0), sem, tid


def counts(pred, xyz, sem, tid):
    gt, tree = sem == 1, tid > 0
    h, _ = ground_surface(xyz)
    base = tree & (h < 2.0)
    ut, tidx = np.unique(np.where(tree, tid, 0), return_inverse=True)
    size = np.bincount(tidx[tree], minlength=len(ut))
    tsize = np.where(ut > 0, size, 0)
    rem = np.bincount(tidx[pred & tree], minlength=len(ut))
    big = tsize >= 50
    return dict(
        tp=int((pred & gt).sum()),
        fp=int((pred & ~gt).sum()),
        fn=int((~pred & gt).sum()),
        tree_pts=int(tree.sum()),
        tree_removed=int((pred & tree).sum()),
        base_pts=int(base.sum()),
        base_removed=int((pred & base).sum()),
        trees=int(big.sum()),
        trees_killed=int((big & (rem >= 0.5 * tsize)).sum()),
    )


def plot_tiles(path):
    t0 = time.time()
    name = os.path.basename(path)[:-4]
    xyz, sem, tid = load(path)
    tij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / TILE).astype(np.int64)
    key = tij[:, 0] * 10_000 + tij[:, 1]
    preds = {m: np.zeros(len(xyz), bool) for m in METHODS}
    routes = []
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        # Too small to be a chunk: left unfiltered.
        if len(idx) < MIN_VOX:
            continue
        masks, cov, _ = filter_tile(xyz[idx])
        routes.append(
            "ss_canupo" if cov >= DEFAULT.coverage_cut else "shape_rule"
        )
        for m in preds:
            preds[m][idx] = masks[m]
    return dict(
        plot=name,
        sensor=sensor(name),
        routes=routes,
        sec=round(time.time() - t0, 1),
        rows={m: counts(p, xyz, sem, tid) for m, p in preds.items()},
    )


def plot_exact(path):
    name = os.path.basename(path)[:-4]
    xyz, sem, tid = load(path)
    rows = {
        "shape_rule": counts(shape_rule(xyz), xyz, sem, tid),
        "ss_canupo": counts(ss_canupo(xyz), xyz, sem, tid),
    }
    return dict(plot=name, sensor=sensor(name), rows=rows)


def line(label, rows):
    t = {k: sum(r[k] for r in rows) for k in KEYS}
    if t["tp"] + t["fn"]:
        P = t["tp"] / max(t["tp"] + t["fp"], 1)
        R = t["tp"] / (t["tp"] + t["fn"])
        F = 2 * P * R / max(P + R, 1e-9)
        prf = f"{100 * P:.1f} | {100 * R:.1f} | {100 * F:.1f}"
    else:
        prf = "– | – | –"
    tree = 100 * t["tree_removed"] / max(t["tree_pts"], 1)
    base = 100 * t["base_removed"] / max(t["base_pts"], 1)
    killed = f"{t['trees_killed']}/{t['trees']}"
    return f"| {label} | {prf} | {tree:.2f} % | {base:.1f} % | {killed} |"


def compare_with_reference(res, method, ref_path, split):
    ref_key = REF_KEYS[method]
    with open(ref_path.replace("{split}", split)) as fh:
        ref = {r["plot"]: r["rows"][ref_key] for r in json.load(fh)}
    same = sum(
        {k: r["rows"][method][k] for k in KEYS}
        == {k: ref[r["plot"]][k] for k in KEYS}
        for r in res
    )
    return (
        f"{split} {method}: per-plot counts identical to the reference "
        f"in {same}/{len(res)} plots"
    )


def routing_lines(res, split):
    routes = [x for r in res for x in r["routes"]]
    n_ss, n_rule = routes.count("ss_canupo"), routes.count("shape_rule")
    by_plot = ", ".join(
        f"{r['plot']} {r['routes'].count('shape_rule')}/{len(r['routes'])}"
        for r in res
        if "shape_rule" in r["routes"]
    )
    return [
        f"{split}: {len(routes)} tiles routed to ss_canupo {n_ss}, "
        f"shape_rule {n_rule}",
        f"  shape_rule tiles by plot: {by_plot}",
        "",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tiles", "exact"], default="tiles")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--ref-rule",
        help="exact mode: per-plot JSON of the original shape-rule run",
    )
    ap.add_argument(
        "--ref-ss",
        help="exact mode: per-plot JSON of the original SS-CANUPO run",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run = plot_tiles if a.mode == "tiles" else plot_exact
    lines = [f"# forest-ground-filter on FOR-instance ({a.mode} mode)", ""]

    for split, sub in (("test", "test_data"), ("val", "train_val_data")):
        pattern = os.path.join(a.root, sub, f"*_{split}.ply")
        # Biggest plots first, so the worker pool finishes evenly.
        files = sorted(glob.glob(pattern), key=lambda p: -os.path.getsize(p))
        with Pool(a.workers) as pool:
            res = sorted(pool.map(run, files), key=lambda r: r["plot"])
        with open(os.path.join(a.out, f"{split}_per_plot.json"), "w") as fh:
            json.dump(res, fh)

        if a.mode == "exact":
            for method, ref in (
                ("shape_rule", a.ref_rule),
                ("ss_canupo", a.ref_ss),
            ):
                if ref:
                    lines.append(
                        compare_with_reference(res, method, ref, split)
                    )
            lines.append("")

        no_bc = [r for r in res if r["sensor"] != "TLS"]
        sets = [
            (f"{split.upper()}, all {len(res)} plots", res),
            (f"{split.upper()}, {len(no_bc)} plots without BlueCat", no_bc),
        ]
        for s in ("ULS", "MLS", "TLS"):
            rs = [r for r in res if r["sensor"] == s]
            sets.append((f"{split.upper()}, {s} ({len(rs)} plots)", rs))
        methods = list(res[0]["rows"])
        for title, rs in sets:
            if rs:
                lines += [f"## {title}", ""] + HEADER
                lines += [line(m, [r["rows"][m] for r in rs]) for m in methods]
                lines.append("")
        if a.mode == "tiles":
            lines += routing_lines(res, split)

    text = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
