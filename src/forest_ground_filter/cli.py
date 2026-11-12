"""Command line: remove ground from every point cloud in a folder.

    forest-ground-filter --in <dir> --out <new dir> [--pattern '*.laz']
                         [--method routed] [--mask-only]
                         [--shard 0/4] [--workers 8]

Output per input file:

- <name>.<ext>: the cloud without ground points, all fields kept
  (skipped with --mask-only).
- <name>.ground.npz: ground mask, one bool per input point in file order
  (read with forest_ground_filter.load_mask).
- One row in stats_shard<i>of<n>.csv: point counts, filter used,
  coverage, seconds.

Files already done are skipped, so runs can be restarted or sharded.
"""

import argparse
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

from .filters import ground_mask
from .io import read_xyz, save_mask, write_kept

FIELDS = [
    "file",
    "n_points",
    "n_ground",
    "n_kept",
    "method",
    "coverage",
    "sec",
]


def _one(job):
    path, out_dir, method, mask_only = job
    name = os.path.basename(path)
    mpath = os.path.join(out_dir, os.path.splitext(name)[0] + ".ground.npz")
    cpath = os.path.join(out_dir, name)
    if os.path.exists(mpath) and (mask_only or os.path.exists(cpath)):
        return None
    t0 = time.time()
    obj, xyz = read_xyz(path)
    g, info = ground_mask(xyz, method=method, return_info=True)
    save_mask(mpath, g)
    if not mask_only:
        write_kept(obj, cpath, ~g)
    return dict(
        file=name,
        n_points=len(g),
        n_ground=int(g.sum()),
        n_kept=int((~g).sum()),
        method=info.get("method", method),
        coverage=info.get("coverage", ""),
        sec=round(time.time() - t0, 2),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Remove ground points from forest point clouds."
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="folder with .laz/.las/.ply files",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="NEW output folder (never the input folder)",
    )
    ap.add_argument("--pattern", default="*.laz")
    ap.add_argument(
        "--method",
        default="routed",
        choices=["routed", "ss_canupo", "shape_rule", "height"],
    )
    ap.add_argument(
        "--mask-only",
        action="store_true",
        help="write only the masks, not the filtered clouds",
    )
    ap.add_argument(
        "--shard", default="0/1", help="i/n: process files k with k %% n == i"
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    if os.path.realpath(a.inp) == os.path.realpath(a.out):
        sys.exit("--out must be a new folder, not the input folder")
    i, n = map(int, a.shard.split("/"))
    files = sorted(glob.glob(os.path.join(a.inp, a.pattern)))[i::n]
    os.makedirs(a.out, exist_ok=True)
    stats = os.path.join(a.out, f"stats_shard{i}of{n}.csv")
    new = not os.path.exists(stats)
    print(
        f"{len(files)} files (shard {i}/{n}), method {a.method}: "
        f"{a.inp} -> {a.out}",
        flush=True,
    )
    t0, done = time.time(), 0
    with open(stats, "a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in pool.imap_unordered(
            _one,
            [(p, a.out, a.method, a.mask_only) for p in files],
            chunksize=4,
        ):
            done += 1
            if r is not None:
                w.writerow(r)
            if done % 200 == 0 or done == len(files):
                fh.flush()
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(files) - done) / max(rate, 1e-6) / 60
                print(
                    f"  {done}/{len(files)}  {rate:.1f} files/s  "
                    f"eta {eta:.0f} min",
                    flush=True,
                )
    print("done", flush=True)


if __name__ == "__main__":
    main()
