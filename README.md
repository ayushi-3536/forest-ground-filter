
# forest-ground-filter: Label-free Ground Removal for Forest Point Clouds

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

</div>

Removes ground points from forest point clouds (TLS, MLS, ULS) without labels or training data, using a self-seeded CANUPO classifier per chunk with a shape-rule fallback for chunks without ground.


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

## Citation

This repository reimplements CANUPO; please cite:

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
```
