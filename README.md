
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
│   ├── methods.py           # All compared filters (CSF / SMRF need CSF and PDAL)
│   ├── forinstance.py       # Benchmark on FOR-instance (ULS, MLS)
│   └── tls_sets.py          # Benchmark on TLS sets
├── examples/
│   └── slurm_array.sbatch
└── tests/
    └── test_synthetic.py
```

## Results

Every plot is cut into 20 m chunks and each chunk is filtered on its own; scores are on 0.05 m voxels. CSF: `cloth-simulation-filter` library defaults (flat-terrain preset, rigidness 3, 1 m cloth). CSF (relief): its relief-terrain preset (rigidness 2) with a 0.5 m cloth. SMRF: PDAL `filters.smrf` defaults (1 m cell).

**Ground F1** (labelled ground):

| Filter | ULS<br>FOR-instance, 27 plots | MLS<br>FOR-instance, 1 plot | TLS<br>ForestSemantic, 1 plot |
|---|---|---|---|
| **SS-CANUPO (routed, default)** | 96.8 | 88.5 | **94.7** |
| Shape rule | **97.0** | 91.7 | 93.7 |
| Height < 0.5 m | 95.4 | 84.0 | 93.8 |
| CSF | 93.6 | 96.7 | 84.3 |
| CSF (relief) | 96.6 | **97.4** | 90.2 |
| SMRF | 96.9 | 95.0 | 94.2 |

**Stem bases removed** (tree points < 2 m above the ground) and **trees killed** (trees losing ≥ 50 % of their points):

| Filter | ULS<br>FOR-instance | MLS<br>FOR-instance | TLS<br>ForestSemantic | TLS<br>TreeScanPL10K, 272 plots | TLS without ground<br>BlueCat | Seconds per chunk |
|---|---|---|---|---|---|---|
| **SS-CANUPO (routed, default)** | **5.3 %** / 0 | 1.8 % / 0 | 7.1 % / 0 | 5.2 % / 0 | **1.0 %** / 0 | 29.5 |
| Shape rule | 10.0 % / 3 | **0.5 %** / 0 | **4.7 %** / 0 | **4.9 %** / 0 | **1.0 %** / 0 | 2.4 |
| Height < 0.5 m | 21.9 % / 2 | 3.7 % / 0 | 12.2 % / 0 | 15.2 % / 0 | 14.1 % / 0 | < 0.1 |
| CSF | 23.1 % / 3 | 2.1 % / 0 | 13.0 % / 0 | 18.3 % / 0 | 19.3 % / 48 | < 0.1 |
| CSF (relief) | 29.5 % / 6 | 2.1 % / 0 | 20.5 % / 0 | 24.1 % / 0 | 22.8 % / 52 | < 0.1 |
| SMRF | 33.5 % / 4 | 3.1 % / 0 | 28.1 % / 0 | 26.8 % / 0 | 16.7 % / 8 | 0.1 |

TreeScanPL10K has no ground labels, so only tree damage is reported. Seconds per chunk: one CPU core, TreeScanPL10K chunks.

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
