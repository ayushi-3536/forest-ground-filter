"""Reading and writing point clouds (.laz, .las, .ply) and ground masks."""
import os

import numpy as np

LAS_EXT = (".laz", ".las")


def read_xyz(path):
    """Read a point cloud. Returns the opened file (keep it for `write_kept`) and its coordinates."""
    if path.lower().endswith(LAS_EXT):
        import laspy
        las = laspy.read(path)
        return las, np.stack([las.x, las.y, las.z], 1)
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    return ply, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def write_kept(obj, path, keep):
    """Write only the points where `keep` is True, with all their original fields.

    We write to a temporary file first and rename it at the end, so a crash never leaves a half-written
    file behind.
    """
    root, ext = os.path.splitext(path)
    tmp = root + ".tmp" + ext  # the temp name must still end in .laz, or laspy writes it uncompressed
    if path.lower().endswith(LAS_EXT):
        import laspy
        laspy.LasData(header=obj.header, points=obj.points[keep]).write(tmp)
    else:
        from plyfile import PlyData, PlyElement
        PlyData([PlyElement.describe(obj["vertex"].data[keep], "vertex")], text=False, byte_order="<").write(tmp)
    os.replace(tmp, path)


def save_mask(path, ground):
    """Save a ground mask (one True/False per point, in file order) as a small compressed file."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ground=np.packbits(ground), n=len(ground))
    os.replace(tmp, path)


def load_mask(path):
    z = np.load(path)
    return np.unpackbits(z["ground"])[: int(z["n"])].astype(bool)
