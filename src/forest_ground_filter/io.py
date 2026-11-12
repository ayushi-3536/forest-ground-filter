"""Reading and writing .laz / .las / .ply point clouds and ground masks."""
import os

import numpy as np

LAS_EXT = (".laz", ".las")


def read_xyz(path):
    """Returns (file object, xyz float64 (n, 3)). The object is passed back to `write_kept`."""
    if path.lower().endswith(LAS_EXT):
        import laspy
        las = laspy.read(path)
        return las, np.stack([las.x, las.y, las.z], 1)
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    return ply, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def write_kept(obj, path, keep):
    """Write the points with keep == True, all point fields preserved. Atomic (temp file + rename)."""
    root, ext = os.path.splitext(path)
    tmp = root + ".tmp" + ext  # keep the extension: laspy chooses LAZ compression from it
    if path.lower().endswith(LAS_EXT):
        import laspy
        laspy.LasData(header=obj.header, points=obj.points[keep]).write(tmp)
    else:
        from plyfile import PlyData, PlyElement
        PlyData([PlyElement.describe(obj["vertex"].data[keep], "vertex")], text=False, byte_order="<").write(tmp)
    os.replace(tmp, path)


def save_mask(path, ground):
    """Bool mask per input point, bit-packed (`ground`, `n`)."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ground=np.packbits(ground), n=len(ground))
    os.replace(tmp, path)


def load_mask(path):
    z = np.load(path)
    return np.unpackbits(z["ground"])[: int(z["n"])].astype(bool)
