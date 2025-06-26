import os
import numpy as np
import torch
import glob

def read_txt_file(txt_path):
    try:
        # Only read XYZ + RGB (6 float columns); skip room name or label
        data = np.loadtxt(txt_path, dtype=np.float32, usecols=range(6))
    except Exception as e:
        print(f"[ERROR] Failed to load {txt_path}: {e}")
        return None, None, None

    coord = data[:, 0:3]
    color = data[:, 3:6] / 255.0  # normalize RGB
    label = np.zeros(data.shape[0], dtype=np.int64)  # placeholder label
    return coord, color, label

def process_area(area_path, save_path, area_name):
    # Use only room files, ignore *_alignmentAngle.txt etc.
    file_list = sorted([
        f for f in glob.glob(os.path.join(area_path, "*.txt"))
        if "alignment" not in f and "pose" not in f and "annotations" not in f
    ])

    print(f"Processing {area_name} with {len(file_list)} room files...")

    all_coords, all_feats, all_labels = [], [], []

    for file_path in file_list:
        coord, color, label = read_txt_file(file_path)
        if coord is None:
            continue
        all_coords.append(coord)
        all_feats.append(color)
        all_labels.append(label)

    if not all_coords:
        print(f"[WARNING] No valid room files found in {area_path}. Skipping.")
        return

    coords = np.concatenate(all_coords, axis=0)
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    data_dict = {
        "coord": coords.astype(np.float32),
        "feat": feats.astype(np.float32),
        "segment": labels.astype(np.int64),
    }

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{area_name}.pth")
    torch.save(data_dict, save_file)
    print(f"[SAVED] {save_file} ({coords.shape[0]} points)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to raw S3DIS (with Area_1/, Area_2/, ...)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save .pth files for training')
    args = parser.parse_args()

    for area in sorted(os.listdir(args.data_root)):
        area_path = os.path.join(args.data_root, area)
        if os.path.isdir(area_path):
            process_area(area_path, args.save_path, area)

if __name__ == "__main__":
    main()

