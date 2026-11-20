<div align="center">

### <img src="./assets/lion.jpg" alt="Image 2" width="4%" style="margin: 0 auto;" > [UniLION: Towards Unified Autonomous Driving Model with Linear Group RNNs](http://arxiv.org/abs/2511.01768)

[Zhe Liu](https://happinesslz.github.io) <sup>1, 2</sup>,
[Jinghua Hou](https://github.com/AlmoonYsl) <sup>1</sup>,
[Xiaoqing Ye](https://shuluoshu.github.io)  <sup>3</sup>,
[Jingdong Wang](https://jingdongwang2017.github.io) <sup>3</sup>,
[Hengshuang Zhao](https://hszhao.github.io) <sup>2,✉</sup>,
[Xiang Bai](https://xbai.vlrlab.net) <sup>1,✉</sup>
<br>
<sup>1</sup> Huazhong University of Science and Technology,
<sup>2</sup> The University of Hong Kong,
<sup>3</sup> Baidu Inc.
<br>
✉ Corresponding author.
<br>

<img src="./assets/arch.png" alt="Image 2" width="100%" style="margin: 0 auto;" >

</div>

## 🔥 Highlights

* **Unified Heterogeneous Inputs**. UniLION integrates multi-view images, LiDAR point clouds, and temporal information into a unified 3D backbone through direct token concatenation without hand-crafted fusion modules.  💪

* **Unified Model**. UniLION enables parameter sharing across different input formats. Specifically, once trained with multi-modal temporal data, the same UniLION model can be directly deployed across different sensor configurations and temporal settings (e.g., LiDAR-only, temporal LiDAR, or multi-modal fusion) without retraining. 💪

* **Unified Representation**. UniLION highly compresses heterogeneous multi-modal and temporal information into a compact BEV feature representation that serves as a shared strong representation for autonomous driving. 💪

* **Strong performance**. UniLION achieves competitive and SOTA performance across comprehensive autonomous driving tasks including 3D perception, motion prediction, and planning. 💪

## News
* **2025.12.15**: [DrivePI](http://arxiv.org/abs/2512.12799) paper released. 🔥
* **2025.12.15**: [GenieDrive](https://github.com/Huster-YZY/GenieDrive) (Physics-Aware Driving World Model) paper released. 🔥
* **2025.06.16**: Our new work about Transformer-Mamba architecture [HybridTM](https://github.com/deepinact/HybridTM) have been accepted by IROS 2025 as **Oral** presentation. 🎉
* **2024.09.26**: LION has been accepted by NeurIPS 2024. 🎉
* **2024.07.25**: LION paper released. 🔥
* **2024.07.02**: Our new works [OPEN](https://github.com/AlmoonYsl/OPEN) and [SEED](https://github.com/happinesslz/SEED) have been accepted by ECCV 2024. 🎉

## Results
* **nuScenes Validation Set**

| Model        | Modality |  NDS |  mAP |  AMOTA   |  mIoU   |  RayIoU   |  minADE (Car/Ped.)  |  L2   |  Col.   | Config | Checkpoint |
|--------------|:--------:|:----:|:----:|:--------:|:-------:|:---------:|:---------:|:-----:|:------:|:------:|:----------:|
|    UniLION   |     L    | 72.3 | 67.5 |   72.6   |   71.7  |   46.8    | - | - | - | - | - |
|    UniLION   |     LT   | 73.0 | 68.9 |   73.3   |   72.4  |   49.6    | 0.58 / 0.39 | 0.60 | 0.27 | - | - |
|    UniLION   |     LC   | 74.9 | 72.2 |   76.2   |   72.3  |   50.8    | - | - | - |[config](https://github.com/happinesslz/UniLION/blob/main/projects/configs/unilion_swin_384_perception.py)| - |
|    UniLION   |     LCT  | 75.4 | 73.2 |   76.5   |   73.3  |   51.3    | 0.57 / 0.37 | 0.65 | 0.18 |[config](projects/configs/unilion_swin_384_seq_e2e.py)| [model](https://drive.google.com/file/d/18fpw-EJ-eJikVPczoRqyhLnXjJzjOnpv/view?usp=drive_link) |

* **3D Object Detection**
  
| Model        | Modality |  NDS |  mAP |
|--------------|:--------:|:----:|:----:|
|    UniLION   |     L    | 72.3 | 67.5 |
|    UniLION   |     LT   | 73.0 | 68.9 |
|    UniLION   |     LC   | 74.9 | 72.2 |
|    UniLION   |     LCT  | 75.4 | 73.2 |

* **Multi-object Tracking**
  
| Model        | Modality |  AMOTA |  AMOTP | IDS |
|--------------|:--------:|:------:|:------:|:---:|
|    UniLION   |     L    | 72.6 | 0.542 | 510 |
|    UniLION   |     LT   | 73.3 | 0.515 | 537 |
|    UniLION   |     LC   | 76.2 | 0.499 | 711 |
|    UniLION   |     LCT  | 76.5 | 0.477 | 613 |

* **BEV Map Segmentation**
  
| Model        | Modality | mIoU |
|--------------|:--------:|:-----:
|    UniLION   |     L    | 71.7 |
|    UniLION   |     LT   | 72.4 |
|    UniLION   |     LC   | 72.3 |
|    UniLION   |     LCT  | 73.3 |

* **3D Occupancy Prediction**
  
| Model        | Modality | RayIoU |
|--------------|:--------:|:-------:
|    UniLION   |     L    |  46.8  |
|    UniLION   |     LT   |  49.6  |
|    UniLION   |     LC   |  50.8  |
|    UniLION   |     LCT  |  51.3  |

* **Motion Prediction**
  
| Model        | Modality | minADE (Car) | minADE (Ped) | EPA |
|--------------|:--------:|:------------:|:------------:|:---:|
|    UniLION   |     LT   | 0.58 | 0.39 | 0.647 |
|    UniLION   |     LCT  | 0.57 | 0.37 | 0.678 |

* **Planning**
  
| Model        | Modality | L2 (1s)	| L2 (2s)	| L2 (3s)	| L2 (avg)	| Col. (1s)	| Col. (2s)	| Col. (3s)	| Col. (avg) |
|--------------|:--------:|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|:---------:|:----------:|
|    UniLION   |     LT   | 0.35 | 0.67 | 1.09 | 0.70 | 0.01 | 0.20 | 0.60 | 0.27 |
|    UniLION   |     LCT  | 0.33 | 0.62 | 0.99 | 0.65 | 0.01 | 0.12 | 0.42 | 0.18 |


## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of UniLION codebase.

# Data Processing
First, download the nuscenes dataset
```
├── can_bus
├── maps
├── occ3d
├── samples
├── sweeps 
└── v1.0-trainval
```

The dataset of occ3d can be downloaded from: https://tsinghua-mars-lab.github.io/Occ3D/

Then, we run the following script:
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --db-save-path ./data/nuscenes/ \
    --version v1.0
```

For planning anchor generation, please run:

```shell script
python kmeans_planning.py
```

Dwonload image pretrained model

We adopt the same image pretrained model with [bevfusion](https://github.com/mit-han-lab/bevfusion/blob/main/tools/download_pretrained.sh). You could download this file in [swint_nuimg_pretrained.pth](https://www.dropbox.com/scl/fi/f3e67wgn2omoftah4ceri/swint-nuimages-pretrained.pth?rlkey=k9kafympye80b3b1quutti4yq)


After completing all the steps above, the final data structure should be organized as follows:
```
├── can_bus
├── kmeans_motion_6.npy
├── kmeans_planning_4096.npy
├── kmeans_planning_6.npy
├── maps
├── nuscenes_dbinfos_train.pkl
├── nuscenes_gt_database
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── occ3d
├── samples
├── sweeps 
└── v1.0-trainval
```



## Getting Started
### Train UniLION on nuScenes

```shell script
# First stage

tools/dist_train.sh projects/unilion_swin_384_det_map.py 8

# Second stage

tools/dist_train.sh projects/unilion_swin_384_seq_perception.py 8

# Third stage

tools/dist_train.sh projects/unilion_swin_384_seq_e2e.py 8
```
Note: for better performance, you can additionly train 3D object detection for providing the pretrained model in the first stage.
```shell script
tools/dist_train.sh projects/unilion_swin_384_det.py 8
```

### Test UniLION on nuScenes
  
You can freely select the tasks to be evaluated in the config.

```shell script
tools/dist_test.sh projects/<CONFIGS> 8 <CKPT> --eval mAP
```

Besides, you can use our released model ([UniLION model](https://drive.google.com/file/d/18fpw-EJ-eJikVPczoRqyhLnXjJzjOnpv/view?usp=drive_link)) to evaluate all results:

```shell script
tools/dist_test.sh projects/projects/configs/unilion_swin_384_seq_e2e.py 8 <CKPT> --eval mAP
```

### Export BEV features without training

Download the released UniLION checkpoint and prepare the nuScenes train/val data
as described above. Then export the fused BEV feature map for each sample; the
script uses the test pipeline for both splits, runs in evaluation mode, and
does not perform training or execute the task heads.

In the `unilion` Conda environment, point CUDA to that environment before
running (adjust the Python site-packages path if using a different Python
version):

```shell script
conda activate unilion
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
```

```shell script
python tools/extract_bev_features.py \
    projects/configs/unilion_swin_384_seq_e2e.py \
    /path/to/unilion_checkpoint.pth \
    --data-root /path/DrivePI_Data/nuscenes \
    --train-output-dir /path/DrivePI_Data/unilion_bev_feats_train \
    --val-output-dir /path/DrivePI_Data/unilion_bev_feats_val \
    --dtype float16
```

Each output directory contains one `<sample_token>.pt` file per sample and a
`manifest.jsonl` index. Each `.pt` file contains a `feature` tensor in `C,H,W`
layout and metadata for its nuScenes sample token, scene, timestamp, BEV
coordinate bounds, and dtype. The default `float16` output reduces storage;
use `--dtype float32` if the downstream project requires full precision. The
checkpoint must be a complete UniLION model checkpoint; no training command is
needed.

The vendored Mamba implementation requires the `causal-conv1d` 1.4.0 CUDA
extension API. The v1.1.1 extension has an incompatible function signature.
Install the matching Git tag from the cloned repository before extraction:

```shell script
cd /path/to/causal-conv1d
git fetch --tags
git checkout v1.4.0
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --force-reinstall .
python -c 'import causal_conv1d_cuda; from importlib.metadata import version; print(version("causal-conv1d"))'
```

### Train UniLION with LiDAR only

These are LiDAR-only versions of the three stages above. Run them in order,
using the same eight-GPU setup assumed by the original configs:

```shell script
# Stage 1: detection and map
tools/dist_train.sh projects/configs/unilion_swin_384_det_map_lidar.py 8 \
    --work-dir work_dirs/unilion_swin_384_det_map_lidar

# Stage 2: sequential perception (loads stage 1's epoch_12.pth)
tools/dist_train.sh projects/configs/unilion_swin_384_seq_perception_lidar.py 8 \
    --work-dir work_dirs/unilion_swin_384_seq_perception_lidar

# Stage 3: end-to-end motion and planning (loads stage 2's iter_42192.pth)
tools/dist_train.sh projects/configs/unilion_swin_384_seq_e2e_lidar.py 8 \
    --work-dir work_dirs/unilion_swin_384_seq_e2e_lidar
```

The configs expect prepared nuScenes files under `/home/renault/repo/dataset/nuscenes/`; edit
`data_root` in each config if needed. Camera feature extraction and image
loading are disabled. Camera calibration metadata is retained only for the
existing LiDAR-to-ego coordinate transform. The standalone
`unilion_swin_384_det_lidar.py` config is a single-stage detection-only option,
not stage 1 of this three-stage recipe.

## TODO
- [x] Release the paper.
- [x] Release the code of UniLION.
- [x] Release checkpoints of UniLION.

## Citation
```
@article{liu2024lion,
  title={LION: Linear Group RNN for 3D Object Detection in Point Clouds},
  author={Zhe Liu, Jinghua Hou, Xingyu Wang, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, Xiang Bai},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{liu2025unilion,
  title={UniLION:Towards Unified Autonomous Driving Model with Linear Group RNNs},
  author={Zhe Liu, Jinghua Hou, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, Xiang Bai},
  journal={arXiv preprint arXiv:2511.01768},
  year={2025}
}
```

## Acknowledgements
We thank these great works and open-source repositories: [LION](https://github.com/happinesslz/LION),
[MMDectection](https://github.com/open-mmlab/mmdetection3d), [SparseDrive](https://github.com/swc-17/SparseDrive), [Mamba](https://github.com/state-spaces/mamba), [RWKV](https://github.com/BlinkDL/RWKV-LM), [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV), and [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention).
