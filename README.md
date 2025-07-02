## SONATA with ZED 2i Camera for Indoor 3D Semantic Segmentation

This project demonstrates the use of Facebook Research's [SONATA](https://github.com/facebookresearch/sonata) pipeline with data captured from the **ZED 2i stereo camera**, processed for indoor **3D semantic segmentation**. The pipeline integrates SONATA’s transformation utilities, preprocessing, and inference modules with custom-collected point clouds, and visualizes results using **Open3D**.

### Hardware

- **Stereo Camera**: ZED 2i (by Stereolabs)
- **GPU**: NVIDIA RTX 3070
- **Capture Mode**: Indoor static scene scanning
- **Point Cloud Format**: `.ply` files exported via ZED SDK

### Software Stack

- **Python 3.8+**
- **Conda Environment** (see below)
- **PyTorch**
- **Open3D**
- **SONATA (FacebookResearch)**
- **ZED SDK (v4.0+)** for point cloud capture

### Conda Environment Setup

The following Conda environment was used for running SONATA:

```yaml
name: sonata_zed
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch=1.13
  - torchvision
  - cudatoolkit=11.3  # match your system's CUDA version
  - open3d
  - numpy
  - matplotlib
  - pip
  - pip:
      - pointops2
      - opencood
````

To create the environment:

```bash
conda env create -f conda_env.yml
conda activate sonata_zed
```

### Scripts Used

#### 1. `preprocess_zed_cloud.py`

Preprocesses the raw `.ply` files from ZED:

* Applies SONATA-style transforms (`CenterShift`, `GridSample`, `ToTensor`, etc.)
* Converts to model-ready numpy format
* Saves intermediate outputs for reproducibility

Run it using:

```bash
python scripts/preprocess_zed_cloud.py --input data/pointclouds/zed_scene1.ply --output data/processed/scene1.npy
```

#### 2. `run_inference.py`

Loads the pretrained SONATA model and runs semantic segmentation:

```bash
python scripts/run_inference.py --input data/processed/scene1.npy --output outputs/segmentation_labels.npy
```

#### 3. `visualize_with_open3d.py`

Visualizes point cloud and segmentation result using Open3D:

```bash
python scripts/visualize_with_open3d.py \
  --cloud data/pointclouds/zed_scene1.ply \
  --labels outputs/segmentation_labels.npy \
  --save outputs/screenshots/zed_scene1_segmented.png
```

### Results

* Successfully ran SONATA's model inference on indoor scenes.
* Segmentation masks align well with object structures from ZED 2i.
* Open3D screenshots capture meaningful label differentiation across surfaces.

### Challenges Faced

* Needed to **manually align** SONATA preprocessing pipeline to raw ZED point cloud format.
* Grid sampling (`GridSample`) caused minor **geometry distortion**, mitigated by tweaking `grid_size`.
* Ensuring **coordinate normalization** between raw `.ply` and SONATA format was non-trivial.

### References

* [SONATA GitHub](https://github.com/facebookresearch/sonata)
* [ZED SDK](https://www.stereolabs.com/docs/)
* [Open3D Docs](http://www.open3d.org/)
