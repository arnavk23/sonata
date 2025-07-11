import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("zed_out.ply")
points = np.asarray(pcd.points)

# Mock segmentation for now (you’ll replace this with model inference)
# Assign random labels for visualization
labels = np.random.randint(0, 5, size=len(points))
colors = np.array([
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [0, 1, 1],  # cyan
])[labels]

# Assign colors
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

