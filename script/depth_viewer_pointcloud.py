import pyzed.sl as sl
import open3d as o3d
import numpy as np

# Initialize camera
zed = sl.Camera()
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.ULTRA
init.coordinate_units = sl.UNIT.METER
zed.open(init)

runtime = sl.RuntimeParameters()
pc = sl.Mat()

if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_measure(pc, sl.MEASURE.XYZRGBA)
    array = pc.get_data()  # (H, W, 4)
    xyz = array[:, :, :3].reshape(-1, 3)
    
    # Remove invalid points
    mask = ~np.isnan(xyz).any(axis=1)
    points = xyz[mask]

    # Save with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("zed_output.ply", pcd)

zed.close()

