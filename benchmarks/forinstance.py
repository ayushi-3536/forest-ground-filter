"""Benchmark on FOR-instance (FOR-instanceV2 layout with BlueCat: <root>/{train_val_data,test_data}/*_{val,test}.ply,
vertex fields x, y, z, semantic_seg (1 = ground), treeID (> 0 = tree)). Labels are used for scoring only.

  --mode tiles   every plot is cut into 20 m tiles, each filtered as an independent chunk (the corpus setting);
                 scores per plot are pooled into the 29-plot (all) and 28-plot (without BlueCat) test sets and per
                 sensor. Methods: routed, ss_canupo, shape_rule, height.
  --mode exact   whole plots, shape_rule and ss_canupo; per-plot counts are compared with reference JSONs
                 (the evaluation the README numbers come from) to show bit-exact reproduction.

Scoring is at 0.05 m voxels (first point per voxel); stem bases = tree points < 2 m above the 2 m-cell minimum;
a tree (>= 50 voxels) is killed if it loses >= 50 % of its voxels.

  python benchmarks/forinstance.py --root <FOR-instance dir> --out <dir> --mode tiles --workers 8
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from plyfile import PlyData

from forest_ground_filter import DEFAULT, ground_mask, shape_rule, ss_canupo
from forest_ground_filter.features import ground_surface, voxel_first

TILE, MIN_VOX = 20.0, 10_000
KEYS = ("tp", "fp", "fn", "tree_pts", "tree_removed", "base_pts", "base_removed", "trees", "trees_killed")


def sensor(name):
    return "TLS" if name.startswith("BlueCat") else "MLS" if name.startswith("NIBIO_MLS") else "ULS"


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
    tsize = np.where(ut > 0, np.bincount(tidx[tree], minlength=len(ut)), 0)
    rem = np.bincount(tidx[pred & tree], minlength=len(ut))
    big = tsize >= 50
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()),
                tree_pts=int(tree.sum()), tree_removed=int((pred & tree).sum()),
                base_pts=int(base.sum()), base_removed=int((pred & base).sum()),
                trees=int(big.sum()), trees_killed=int((big & (rem >= 0.5 * tsize)).sum()))


def plot_tiles(path):
    t0 = time.time()
    name = os.path.basename(path)[:-4]
    xyz, sem, tid = load(path)
    tij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / TILE).astype(np.int64)
    key = tij[:, 0] * 10_000 + tij[:, 1]
    preds = {m: np.zeros(len(xyz), bool) for m in ("routed", "ss_canupo", "shape_rule", "height")}
    routes = []
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        if len(idx) < MIN_VOX:  # too small to be a chunk: left unfiltered
            continue
        for m in preds:
            if m == "routed":
                g, info = ground_mask(xyz[idx], method=m, return_info=True)
                routes.append(info["method"])
            else:
                g = ground_mask(xyz[idx], method=m)
            preds[m][idx] = g
    return dict(plot=name, sensor=sensor(name), routes=routes, sec=round(time.time() - t0, 1),
                rows={m: counts(p, xyz, sem, tid) for m, p in preds.items()})


def plot_exact(path):
    name = os.path.basename(path)[:-4]
    xyz, sem, tid = load(path)
    return dict(plot=name, sensor=sensor(name),
                rows={"shape_rule": counts(shape_rule(xyz), xyz, sem, tid), "ss_canupo": counts(ss_canupo(xyz), xyz, sem, tid)})


def line(label, rows):
    t = {k: sum(r[k] for r in rows) for k in KEYS}
    if t["tp"] + t["fn"]:
        P, R = t["tp"] / max(t["tp"] + t["fp"], 1), t["tp"] / (t["tp"] + t["fn"])
        prf = f"{100 * P:.1f} | {100 * R:.1f} | {200 * P * R / max(P + R, 1e-9):.1f}"
    else:
        prf = "– | – | –"
    return (f"| {label} | {prf} | {100 * t['tree_removed'] / max(t['tree_pts'], 1):.2f} % | "
            f"{100 * t['base_removed'] / max(t['base_pts'], 1):.1f} % | {t['trees_killed']}/{t['trees']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tiles", "exact"], default="tiles")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ref-rule", help="exact mode: per-plot JSON with rows[RULE_KEY] (eval_ground_filters output)")
    ap.add_argument("--ref-ss", help="exact mode: per-plot JSON with rows[SS_KEY] (canupo_lda output)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = [f"# forest-ground-filter on FOR-instance ({a.mode} mode)", ""]
    hdr = ["| Filter | P | R | F1 | tree pts removed | stem bases removed | trees killed |", "|---|---|---|---|---|---|---|"]
    for split, sub in (("test", "test_data"), ("val", "train_val_data")):
        files = sorted(glob.glob(os.path.join(a.root, sub, f"*_{split}.ply")), key=lambda p: -os.path.getsize(p))
        with Pool(a.workers) as pool:
            res = sorted(pool.map(plot_tiles if a.mode == "tiles" else plot_exact, files), key=lambda r: r["plot"])
        json.dump(res, open(os.path.join(a.out, f"{split}_per_plot.json"), "w"))
        methods = list(res[0]["rows"])
        if a.mode == "exact":
            for m, ref, key in (("shape_rule", a.ref_rule, "canupo_s1.2_p0.3_a20_h1.5_rel"), ("ss_canupo", a.ref_ss, "canupo_lda")):
                if ref:
                    rr = {r["plot"]: r["rows"][key] for r in json.load(open(ref.replace("{split}", split)))}
                    same = sum({k: r["rows"][m][k] for k in KEYS} == {k: rr[r["plot"]][k] for k in KEYS} for r in res)
                    L.append(f"{split} {m}: per-plot counts identical to the reference in {same}/{len(res)} plots")
            L.append("")
        sets = [(f"{split.upper()}, all {len(res)} plots", res),
                (f"{split.upper()}, {sum(r['sensor'] != 'TLS' for r in res)} plots without BlueCat", [r for r in res if r["sensor"] != "TLS"])]
        sets += [(f"{split.upper()}, {s} ({sum(r['sensor'] == s for r in res)} plots)", [r for r in res if r["sensor"] == s])
                 for s in ("ULS", "MLS", "TLS")]
        for title, rs in sets:
            if rs:
                L += [f"## {title}", ""] + hdr + [line(m, [r["rows"][m] for r in rs]) for m in methods] + [""]
        if a.mode == "tiles":
            rt = [x for r in res for x in r["routes"]]
            L += [f"{split}: {len(rt)} tiles routed to ss_canupo {rt.count('ss_canupo')}, shape_rule {rt.count('shape_rule')}"]
            L += ["  shape_rule tiles by plot: " + ", ".join(f"{r['plot']} {r['routes'].count('shape_rule')}/{len(r['routes'])}"
                                                         for r in res if 'shape_rule' in r["routes"]), ""]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
