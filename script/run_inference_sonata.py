import torch
import numpy as np
import os
from sonata.config import get_cfg
from sonata.models.build import build_model
from sonata.utils.checkpoint import load_checkpoint

def run_inference(cfg_path, input_path, model_path, output_path):
    # Load config
    from sonata.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    # Load input .npz file
    data = np.load(input_path)
    xyz = data['xyz']
    rgb = data['rgb']

    coords = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    feats = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)   # (1, N, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg)
    model.to(device)
    model.eval()

    load_checkpoint(model, model_path, strict=True, logger=None)

    with torch.no_grad():
        pred = model(feats.to(device), coords.to(device))
        pred_labels = pred.argmax(dim=-1).squeeze(0).cpu().numpy()

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, os.path.basename(input_path).replace(".npz", "_pred.npz"))
    np.savez(save_path, pred=pred_labels)
    print(f"Saved predictions to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--input", required=True, help="Path to input .npz file")
    parser.add_argument("--weights", required=True, help="Path to pretrained SONATA model")
    parser.add_argument("--output", default="output", help="Directory to save predictions")

    args = parser.parse_args()
    run_inference(args.cfg, args.input, args.weights, args.output)

