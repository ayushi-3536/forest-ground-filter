"""Reading and writing point clouds (.laz, .las, .ply) and ground masks."""

import os

import numpy as np

LAS_EXT = (".laz", ".las")


def read_xyz(path):
    """Read a point cloud.

    Returns:
        obj: the opened file, to pass to `write_kept`.
        xyz: (n, 3) float64 coordinates.
    """
    if path.lower().endswith(LAS_EXT):
        import laspy

        las = laspy.read(path)
        return las, np.stack([las.x, las.y, las.z], 1)
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"]
    return ply, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def write_kept(obj, path, keep):
    """Write the points where `keep` is True, with all their fields.

    Writes to a temporary file and renames it, so a crash never leaves a
    partial file.
    """
    root, ext = os.path.splitext(path)
    # Keep the extension: laspy picks LAZ compression from it.
    tmp = root + ".tmp" + ext
    if path.lower().endswith(LAS_EXT):
        import laspy

        laspy.LasData(header=obj.header, points=obj.points[keep]).write(tmp)
    else:
        from plyfile import PlyData, PlyElement

        PlyData(
            [PlyElement.describe(obj["vertex"].data[keep], "vertex")],
            text=False,
            byte_order="<",
        ).write(tmp)
    os.replace(tmp, path)


def save_mask(path, ground):
    """Save a ground mask (one bool per point, in file order), bit-packed."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ground=np.packbits(ground), n=len(ground))
    os.replace(tmp, path)


def load_mask(path):
    """Load a mask written by `save_mask`."""
    z = np.load(path)
    return np.unpackbits(z["ground"])[: int(z["n"])].astype(bool)
