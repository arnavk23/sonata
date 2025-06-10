import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import sonata

try:
    import flash_attn
except ImportError:
    flash_attn = None

# ScanNet class IDs and color map
VALID_CLASS_IDS_20 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    16: (219.0, 219.0, 141.0),
    24: (255.0, 127.0, 14.0),
    28: (158.0, 218.0, 229.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    36: (227.0, 119.0, 194.0),
    39: (82.0, 84.0, 163.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def main():
    sonata.utils.set_seed(24525867)

    # Load your ZED point cloud
    pcd = o3d.io.read_point_cloud("zed_out.ply")
    xyz = np.asarray(pcd.points).astype(np.float32)
    rgb = np.asarray(pcd.colors).astype(np.float32)

    # SONATA expects a dict with keys: coord, feat
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    normals = np.asarray(pcd.normals).astype(np.float32)

    point = {
            "coord": xyz,
            "color": rgb,
            "normal": normals,
    }

    print("Loaded point cloud:")
    print(" - Shape:", point["coord"].shape)
    print(" - Sample coords:\n", point["coord"][:5])

    # Load model
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()

    # Load segmentation head
    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])

    # Apply default SONATA transforms
    transform = sonata.transform.default()
    point = transform(point)

    model.eval()
    seg_head.eval()

    with torch.inference_mode():
        for k, v in point.items():
            if isinstance(v, torch.Tensor):
                point[k] = v.cuda(non_blocking=True)

        # Forward through backbone
        point = model(point)

        # Reverse pooling (multi-stage decoder)
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).cpu().numpy()

        # Map class IDs to RGB colors
        colors = np.array(CLASS_COLOR_20)[pred]

    # Visualize results
    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = o3d.utility.Vector3dVector(xyz)
    pcd_result.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.visualization.draw_geometries([pcd_result])


if __name__ == "__main__":
    main()

