
# forest-ground-filter: Label-free Ground Removal for Forest Point Clouds

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

</div>

Removes ground points from forest point clouds (TLS, MLS, ULS) without labels or training data, using a self-seeded, CANUPO-style classifier per chunk with a shape-rule fallback for chunks without ground.


## Setup

```
git clone https://github.com/GeoSense-Freiburg/forest-ground-filter
cd forest-ground-filter
pip install .
```
---

## Usage

```python
from forest_ground_filter import ground_mask

g = ground_mask(xyz)   # xyz: (n, 3) array in metres -> bool per point, True = ground
```

From the command line, for a folder of `.laz` / `.las` / `.ply` files:

```bash
# ground-free copies + ground masks
forest-ground-filter --in chunks/ --out chunks_noground/

# masks only, split into 8 shards
forest-ground-filter --in chunks/ --out ground_masks/ --mask-only --shard 0/8
```

For a Slurm array, see [`examples/slurm_array.sbatch`](./examples/slurm_array.sbatch).

## Filters

Every filter first thins the chunk to one point per 5 cm voxel and measures each point's height above the local floor (the lowest point of its 2 m × 2 m column). Choose one with `ground_mask(xyz, method=...)`.

**Height** (`"height"`)
- Ground = every point less than 0.5 m above the local floor.
- Simple and fast, but also removes the lowest 0.5 m of every stem and shrub.

**Shape rule** (`"shape_rule"`)
- Look at each low point's neighbours within 1.2 m.
- Ground = the neighbours form a flat sheet, the sheet is tilted less than 20° from the terrain, and the point is less than 1.5 m above the floor.
- Points with too few neighbours to judge: ground if below 0.5 m.
- Stems and shrubs are rarely flat, so few trees are removed.

**SS-CANUPO, self-seeded CANUPO** (`"ss_canupo"`)
- Take the lowest points of the chunk (less than 10 cm above the floor) as ground examples, and points more than 3 m up as vegetation examples.
- Describe every point by how line-like and how plane-like its neighbourhood is at 10 radii, from 0.1 m to 2 m (multi-scale dimensionality, as in CANUPO).
- Train a linear classifier on the examples of this chunk only, so it adapts to each scanner and density.
- Classify every point below 1.5 m with it.
- Removes fewer stem bases than the shape rule, but fails when the lowest points are not ground (e.g. scans with the ground already cut away).

**Routed** (`"routed"`, default)
- Run the shape rule and count in how many 2 m columns it finds ground (coverage).
- Coverage ≥ 0.63: the chunk has a floor, so use SS-CANUPO.
- Coverage < 0.63: the chunk has no floor, so keep the shape rule's result.

## Repository Structure

```text
forest-ground-filter/
├── src/forest_ground_filter/
│   ├── filters.py           # Ground filters, settings, ground_mask()
│   ├── features.py          # Voxels, height above ground, multi-scale features, LDA
│   ├── io.py                # Read / write point clouds and masks
│   └── cli.py               # Command line
├── benchmarks/
│   └── forinstance.py       # Benchmark on FOR-instance
├── examples/
│   └── slurm_array.sbatch
└── tests/
    └── test_synthetic.py
```

## Results

RESULTS_TABLES

## Acknowledgements

- SS-CANUPO builds on **CANUPO** (Brodu & Lague, 2012): classifying points by their neighbourhood dimensionality at several scales with a linear classifier trained per scene. It is not a reimplementation of the CANUPO software.
- The dimensionality features (a1D, a2D) follow **Demantké et al. (2011)**.

## Related ground filters

Established ground filters, mostly developed for airborne lidar:

- Axelsson (2000): progressive TIN densification. *Int. Arch. Photogramm. Remote Sens.* 33(B4), 110–117.
- Zhang et al. (2003): progressive morphological filter. *IEEE TGRS* 41(4), 872–882. doi:10.1109/TGRS.2003.810682
- Pingel et al. (2013): simple morphological filter (SMRF). *ISPRS J.* 77, 21–30. doi:10.1016/j.isprsjprs.2012.12.002
- Zhang et al. (2016): cloth simulation filter (CSF). *Remote Sensing* 8(6), 501. doi:10.3390/rs8060501

## Citation

If you use this repository, please cite:

```bibtex
@software{sharma2026forestgroundfilter,
      title={forest-ground-filter: Label-free Ground Removal for Forest Point Clouds},
      author={Ayushi Sharma},
      year={2026},
      version={0.1.0},
      url={https://github.com/GeoSense-Freiburg/forest-ground-filter},
}
```

and the works it builds on:

```bibtex
@article{brodu2012canupo,
      title={3D terrestrial lidar data classification of complex natural scenes using a multi-scale dimensionality criterion: Applications in geomorphology},
      author={Nicolas Brodu and Dimitri Lague},
      journal={ISPRS Journal of Photogrammetry and Remote Sensing},
      volume={68},
      pages={121--134},
      year={2012},
      doi={10.1016/j.isprsjprs.2012.01.006},
}

@inproceedings{demantke2011dimensionality,
      title={Dimensionality based scale selection in 3D lidar point clouds},
      author={J{\'e}r{\^o}me Demantk{\'e} and Cl{\'e}ment Mallet and Nicolas David and Bruno Vallet},
      booktitle={International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
      volume={XXXVIII-5/W12},
      pages={97--102},
      year={2011},
      doi={10.5194/isprsarchives-XXXVIII-5-W12-97-2011},
}
```
