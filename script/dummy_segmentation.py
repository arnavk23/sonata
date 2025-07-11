import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def euclidean_clustering(pcd, eps=0.02, min_points=50):
    # DBSCAN clustering from Open3D (approximates Euclidean clustering)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()

    print(f"Point cloud has {max_label + 1} clusters")

    # Assign unique color per cluster
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label >= 0 else 1))
    colors[labels < 0] = [0, 0, 0, 1]  # Noise points in black
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    plane = pcd.select_by_index(inliers)
    rest = pcd.select_by_index(inliers, invert=True)

    return pcd, labels

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("zed_output.ply")
    clustered_pcd, labels = euclidean_clustering(pcd, eps=0.03, min_points=30)
    o3d.visualization.draw_geometries([clustered_pcd], window_name="Instance Segmentation")


