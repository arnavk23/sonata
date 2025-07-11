import open3d as o3d
import numpy as np
import torch
from open3d.ml.torch.models import RandLANet
from open3d.ml.torch.pipelines import SemanticSegmentation
import matplotlib.pyplot as plt

# Generate dummy point cloud with colors and save as PLY
points = np.random.rand(1000, 3).astype(np.float32)  # Nx3 points
colors = np.random.rand(1000, 3).astype(np.float32)  # Nx3 colors in [0,1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("zed_output.ply", pcd)

def main():
    # Load the point cloud
    pcd = o3d.io.read_point_cloud("zed_output.ply")
    points = np.asarray(pcd.points).astype(np.float32)
    
    # Dummy labels to avoid KeyError
    dummy_labels = np.zeros(points.shape[0], dtype=np.int64)

    # Initialize model and pipeline
    model = RandLANet(num_classes=19)
    pipeline = SemanticSegmentation(model=model, device='cpu')

    # Use 'point' key (singular), as expected by RandLANet preprocess
    data = {
        'point': points,
        'label': dummy_labels,
    }

    # Run inference
    result = pipeline.run_inference(data)

    # Get predicted labels and color map
    labels = result['predict_labels']
    max_label = labels.max()
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))[:, :3]

    # Assign colors and visualize
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Semantic Segmentation")

if __name__ == "__main__":
    main()

