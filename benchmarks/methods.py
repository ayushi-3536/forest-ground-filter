"""All filters compared in the benchmarks, applied to one tile.

Ours (forest_ground_filter): routed, ss_canupo, shape_rule, height.

Baselines, used only if their packages import (no tuning on our labels):

- csf_default: cloth simulation filter (Zhang et al. 2016), library
  defaults.
- csf_relief: CSF with rigidness 2 (the setting its authors suggest for
  relief terrain), cloth resolution 0.5 m, slope smoothing on.
- smrf_default: simple morphological filter (Pingel et al. 2013), PDAL
  defaults.
- smrf_cell05: SMRF with a 0.5 m cell instead of 1 m, for dense scans.

Every filter sees the same input: the tile thinned to 0.05 m voxels and
centred. Its answer is copied back to all points of each voxel.
"""

import json
import time

import numpy as np

from forest_ground_filter import DEFAULT, coverage, shape_rule, ss_canupo
from forest_ground_filter.features import voxel_first
from forest_ground_filter.filters import height

try:
    import CSF
except ImportError:
    CSF = None
try:
    import pdal
except ImportError:
    pdal = None

OURS = ("routed", "ss_canupo", "shape_rule", "height")
BASELINES = tuple(
    (["csf_default", "csf_relief"] if CSF is not None else [])
    + (["smrf_default", "smrf_cell05"] if pdal is not None else [])
)
METHODS = OURS + BASELINES


def csf_mask(v, **params):
    """Ground mask from the cloth simulation filter."""
    f = CSF.CSF()
    for k, val in params.items():
        setattr(f.params, k, val)
    f.setPointCloud(np.ascontiguousarray(v, dtype=np.float64))
    ground, other = CSF.VecInt(), CSF.VecInt()
    f.do_filtering(ground, other, exportCloth=False)
    g = np.zeros(len(v), bool)
    g[np.asarray(ground, np.int64)] = True
    return g


def smrf_mask(v, **params):
    """Ground mask from PDAL's simple morphological filter."""
    arr = np.zeros(len(v), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    arr["X"], arr["Y"], arr["Z"] = v[:, 0], v[:, 1], v[:, 2]
    stage = {"type": "filters.smrf", **params}
    p = pdal.Pipeline(json.dumps([stage]), arrays=[arr])
    p.execute()
    out = p.arrays[0]
    assert len(out) == len(v), (len(out), len(v))
    return out["Classification"] == 2


BASELINE_FUNCS = {
    "csf_default": lambda v: csf_mask(v),
    "csf_relief": lambda v: csf_mask(
        v, rigidness=2, cloth_resolution=0.5, bSloopSmooth=True
    ),
    "smrf_default": lambda v: smrf_mask(v),
    "smrf_cell05": lambda v: smrf_mask(v, cell=0.5),
}


def filter_tile(xyz):
    """Run every available filter on one tile.

    Returns:
        masks: {method: bool per input point}.
        cov: the routing coverage of the tile.
        sec: {method: seconds}.
    """
    s = DEFAULT
    first, inv = voxel_first(xyz, s.grid)
    v = xyz[first]
    v = v - v.mean(0)
    sec = {}

    t = time.time()
    rule = shape_rule(v, s)
    sec["shape_rule"] = time.time() - t
    cov = coverage(v, rule, s)
    t = time.time()
    ss = ss_canupo(v, s)
    sec["ss_canupo"] = time.time() - t
    sec["routed"] = sec["shape_rule"] + (
        sec["ss_canupo"] if cov >= s.coverage_cut else 0.0
    )
    t = time.time()
    hr = height(v, s)
    sec["height"] = time.time() - t
    out = {
        "shape_rule": rule,
        "ss_canupo": ss,
        "height": hr,
        "routed": ss if cov >= s.coverage_cut else rule,
    }
    for m in BASELINES:
        t = time.time()
        out[m] = BASELINE_FUNCS[m](v)
        sec[m] = time.time() - t
    return {m: g[inv] for m, g in out.items()}, cov, sec
