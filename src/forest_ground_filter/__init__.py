"""forest-ground-filter: label-free ground removal for forest point clouds.

    from forest_ground_filter import ground_mask
    g = ground_mask(xyz)          # bool per point, True = ground; xyz: (n, 3) array in metres

Default method: routed (self-seeded CANUPO where the chunk has ground, the shape rule elsewhere).
"""
from .filters import DEFAULT, Settings, coverage, ground_mask, routed, shape_rule, ss_canupo
from .io import load_mask, read_xyz, save_mask, write_kept

__version__ = "0.1.0"
__all__ = ["ground_mask", "Settings", "DEFAULT", "routed", "ss_canupo", "shape_rule", "coverage",
           "read_xyz", "write_kept", "save_mask", "load_mask"]
