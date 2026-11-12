# forest-ground-filter

Label-free ground removal for forest point clouds (TLS, MLS, ULS), built for large crowd-sourced corpora where
sensors, densities and sites vary and no labels are available.

```python
from forest_ground_filter import ground_mask
g = ground_mask(xyz)            # xyz: (n, 3) array in metres; g: bool per point, True = ground
```

```bash
pip install .
forest-ground-filter --in chunks/ --out chunks_noground/ --pattern '*.laz'      # ground-free copies + masks
forest-ground-filter --in chunks/ --out ground_masks/ --mask-only --shard 0/8    # masks only, 1 of 8 shards
```

## Method

Per chunk (any size; we use 20 x 20 m), with no training data and fixed settings:

1. **Voxelise** at 0.05 m (first point per voxel); every input point later gets its voxel's label.
2. **Height above the local floor:** the chunk is cut into 2 m cells; height = z minus the lowest z of the cell.
3. **Shape rule** (safe first pass): a point is ground if its neighbourhood within 1.2 m is planar
   (a2D = (s2 - s3) / s1 >= 0.3 from local PCA), faces the local terrain (normal within 20 deg of the terrain
   normal) and is low (< 1.5 m). Points with < 5 neighbours: height < 0.5 m.
4. **Does the chunk have ground?** Coverage = share of occupied 2 m cells in which the shape rule finds ground.
   A real floor appears almost everywhere; a ground-free chunk (pre-cleaned scans, canopy-only tiles) only
   gets scattered hits.
5. **Route:** coverage < 0.63 -> keep the shape rule's result. Otherwise:
6. **Self-seeded CANUPO (SS-CANUPO).** CANUPO (Brodu & Lague 2012) describes every point by its multi-scale
   dimensionality -- a1D = (s1 - s2) / s1 and a2D at 10 radii from 0.1 to 2 m -- and separates two classes with a
   linear discriminant (LDA). CANUPO's training samples are normally picked by hand; here every chunk seeds
   itself: points < 0.10 m above the cell minimum are ground samples, points > 3 m up are vegetation samples
   (20,000 each). The chunk's own LDA then classifies every point below 1.5 m by shape, so stem bases (line- or
   volume-like) are kept even though they are low. Training per chunk adapts the classifier to each sensor and
   density. Step 4 makes sure the seeds are only used where the bottom layer really is ground.

All settings are the fields of `forest_ground_filter.Settings`; `ground_mask(xyz, method=...)` also exposes
the single filters (`"ss_canupo"`, `"shape_rule"`, `"height"`).

## Results

FOR-instance test (labels used for scoring only), 0.05 m voxels, every plot cut into 20 m tiles that are
filtered independently. *Stem bases* = tree points < 2 m above the cell minimum; a tree is *killed* if it loses
>= 50 % of its points.

RESULTS_TABLES

## Limitations

- Steep or rocky terrain is not covered by the benchmark (steepest FOR-instance test plot ~17 deg).
- The coverage cut-off (0.63) is the middle of the gap seen on labelled FOR-instance tiles (no-ground max 0.49,
  ground min 0.77); FOR-instance has no partial tiles (some ground, mostly canopy), and its no-ground tiles are
  one TLS site (BlueCat) plus canopy-only edge tiles.
- Flat low vegetation (moss, dwarf shrubs) is planar and tends to be removed with the ground.
- Very sparse ground (< 5 points within 1 m) falls back to a height rule.

## Reproducing the numbers

`benchmarks/forinstance.py --mode tiles` produces the tables above; `--mode exact` shows that the shape rule and
SS-CANUPO reproduce the per-plot counts of the original evaluation exactly. `tests/` runs a synthetic check
(`pytest`).

## Citation

If you use this, please cite CANUPO:

N. Brodu and D. Lague (2012). 3D terrestrial lidar data classification of complex natural scenes using a
multi-scale dimensionality criterion: applications in geomorphology. ISPRS Journal of Photogrammetry and Remote
Sensing 68, 121-134. doi:10.1016/j.isprsjprs.2012.01.006

This repository reimplements the CANUPO descriptor and classifier from the paper; it contains no CANUPO code.

## Licence

Apache-2.0, see `LICENSE`.
