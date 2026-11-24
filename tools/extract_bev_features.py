import argparse
import copy
import importlib
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import import_modules_from_strings
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from torch import nn
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.configs.lidar_only_utils import make_lidar_only_pipeline  # noqa: E402

# Same model overrides as the *_lidar.py configs.
LIDAR_ONLY_MODEL_OVERRIDES = dict(
    fusion=False,
    with_depth_loss=False,
    img_backbone=None,
    img_neck=None,
    img_map2bev=None,
    use_grid_mask=False,
)


class SceneOrderedSampler(Sampler):
    """Keep each scene contiguous and chronological for temporal inference."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = sorted(
            range(len(dataset)),
            key=lambda index: (
                dataset.data_infos[index]["scene_token"],
                dataset.data_infos[index]["timestamp"],
            ),
        )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class FeatureOnlyModel(nn.Module):
    """Run UniLION's feature extractor without executing task heads."""

    def __init__(self, detector, use_img):
        super().__init__()
        self.detector = detector
        self.use_img = use_img

    def forward(self, points, img=None, **data):
        data.pop("return_loss", None)
        data.pop("rescale", None)
        # MultiScaleFlipAug3D wraps every key in a per-aug list.
        if isinstance(points[0], list):
            points = points[0]
            img = img[0] if img is not None else None
            data = {
                key: value[0] if isinstance(value, list) else value
                for key, value in data.items()
            }
        if not self.use_img:
            img = None
        elif img is None:
            raise RuntimeError(
                "Fusion mode needs 'img' in the test pipeline Collect3D keys."
            )
        return self.detector.extract_feat(points, img, **data)[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export UniLION BEV features for nuScenes train/val."
    )
    parser.add_argument("config", help="UniLION model config")
    parser.add_argument("checkpoint", help="Pretrained UniLION checkpoint")
    parser.add_argument("--data-root", help="nuScenes root (overrides config)")
    parser.add_argument(
        "--train-output-dir",
        required=True,
        help="Directory for train features and manifest",
    )
    parser.add_argument(
        "--val-output-dir",
        required=True,
        help="Directory for val features and manifest",
    )
    parser.add_argument(
        "--modality",
        choices=("auto", "lidar", "fusion"),
        default="auto",
        help="auto: fusion if the config has an img_backbone, else lidar. "
        "lidar: drop the image branch and image pipeline steps. "
        "fusion: feed images (config must define the image branch).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Saved feature dtype; float16 reduces disk usage",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed (~3x larger files, "
        "faster write)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-sample features and manifests",
    )
    return parser.parse_args()


def import_config_plugins(cfg):
    if cfg.get("custom_imports"):
        print("Importing custom modules...")
        import_modules_from_strings(**cfg["custom_imports"])
    if cfg.get("plugin", False):
        print("Importing plugin module...")
        plugin_module = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        importlib.import_module(plugin_module.strip("/").replace("/", "."))


def resolve_modality(cfg, modality):
    has_img_branch = cfg.model.get("img_backbone") is not None
    if modality == "auto":
        modality = "fusion" if has_img_branch else "lidar"
    if modality == "fusion" and not has_img_branch:
        raise ValueError("--modality fusion requires a config with img_backbone.")
    if modality == "lidar" and has_img_branch:
        print(
            "WARNING: lidar-only extraction from a fusion config; the image "
            "branch is disabled and its checkpoint weights are ignored."
        )
        cfg.model.update(LIDAR_ONLY_MODEL_OVERRIDES)
    return modality


def make_dataset_loader(cfg, split, data_root, workers, modality):
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    split_cfg = cfg.data[split]
    # Unwrap dataset wrappers (e.g. CBGSDataset for train).
    while "dataset" in split_cfg:
        split_cfg = split_cfg.dataset
    dataset_cfg.ann_file = split_cfg.ann_file
    if data_root:
        dataset_cfg.data_root = data_root
        dataset_cfg.ann_file = os.path.join(
            data_root, f"nuscenes_infos_{split}.pkl"
        )
    if modality == "lidar":
        dataset_cfg.pipeline = make_lidar_only_pipeline(
            dataset_cfg.pipeline, dataset_cfg.data_root
        )

    dataset = build_dataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SceneOrderedSampler(dataset),
        num_workers=workers,
        collate_fn=partial(collate, samples_per_gpu=1),
        pin_memory=False,
    )
    return dataset, loader


def unwrap_single_meta(value):
    while True:
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    if not isinstance(value, dict):
        raise RuntimeError("Expected exactly one sample's image metadata.")
    return value


def save_npz(feature_path, feature, compress):
    """Save as DrivePI expects: np.load(path)['data'] of shape (1, C, H, W)."""
    tmp_path = feature_path.with_name(feature_path.name + ".tmp")
    save = np.savez_compressed if compress else np.savez
    # Write through a file object so numpy does not append '.npz' to the tmp
    # name, then rename so an interrupted run never leaves a truncated .npz.
    with tmp_path.open("wb") as file:
        save(file, data=feature.unsqueeze(0).numpy())
    os.replace(tmp_path, feature_path)


def export_split(
    model, cfg, split, dataset, data_loader, output_dir, dtype, overwrite,
    compress,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --overwrite to replace it."
        )

    range_values = cfg.model.pc_range
    x_min, y_min, _, x_max, y_max, _ = range_values
    save_dtype = torch.float16 if dtype == "float16" else torch.float32

    model.module.detector.reset()
    model.eval()
    manifest_mode = "w" if overwrite else "x"
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        for index, batch in enumerate(data_loader):
            meta = unwrap_single_meta(batch["img_metas"])
            with torch.no_grad():
                feature_output = model(
                    return_loss=False, rescale=True, **batch
                )

            if isinstance(feature_output, (tuple, list)):
                if len(feature_output) != 1:
                    raise RuntimeError(
                        "Expected UniLION pts_neck to return one fused BEV map, "
                        f"got {len(feature_output)}."
                    )
                feature_output = feature_output[0]
            if (
                not isinstance(feature_output, torch.Tensor)
                or feature_output.ndim != 4
                or feature_output.shape[0] != 1
            ):
                raise RuntimeError(
                    "Expected BEV features with shape (1, C, H, W), "
                    f"got {type(feature_output).__name__} "
                    f"{getattr(feature_output, 'shape', None)}."
                )

            feature = feature_output[0].detach().to(
                device="cpu", dtype=save_dtype
            ).contiguous()
            sample_token = meta["token"]
            scene_token = meta["scene_token"]
            feature_name = f"{sample_token}.npz"
            feature_path = output_dir / feature_name
            if feature_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{feature_path} already exists; pass --overwrite to replace it."
                )

            save_npz(feature_path, feature, compress)

            # The .npz only holds 'data'; per-sample metadata lives here.
            height, width = feature.shape[-2:]
            manifest.write(
                json.dumps(
                    {
                        "schema": "unilion_bev_feature_npz_v1",
                        "split": split,
                        "sample_token": sample_token,
                        "scene_token": scene_token,
                        "timestamp": float(meta["timestamp"]),
                        "file": feature_name,
                        "npz_key": "data",
                        "layout": "1,C,H,W",
                        "shape": [1] + list(feature.shape),
                        "dtype": dtype,
                        "coordinate_frame": "current_ego",
                        "bev_xy_bounds_m": [
                            [float(x_min), float(x_max)],
                            [float(y_min), float(y_max)],
                        ],
                        "meters_per_cell_xy": [
                            float(x_max - x_min) / width,
                            float(y_max - y_min) / height,
                        ],
                    }
                )
                + "\n"
            )
            manifest.flush()

            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"{split}: exported {index + 1}/{len(dataset)} samples")


def main():
    print("Starting UniLION feature extraction...")
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("UniLION feature extraction requires a CUDA GPU.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.gpu_id} is not available.")
    print(f"Using CUDA device {args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    cfg = Config.fromfile(args.config)
    import_config_plugins(cfg)
    print(f"Loaded config from {args.config}")
    modality = resolve_modality(cfg, args.modality)
    print(f"Modality: {modality}")

    print("Setting model configuration...")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get("img_backbone") is not None:
        cfg.model.img_backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    feature_model = MMDataParallel(
        FeatureOnlyModel(model, use_img=modality == "fusion").cuda(args.gpu_id),
        device_ids=[args.gpu_id],
    )

    for split, output_dir in (
        ("train", args.train_output_dir),
        ("val", args.val_output_dir),
    ):
        dataset, data_loader = make_dataset_loader(
            cfg, split, args.data_root, args.workers, modality
        )
        export_split(
            feature_model,
            cfg,
            split,
            dataset,
            data_loader,
            output_dir,
            args.dtype,
            args.overwrite,
            not args.no_compress,
        )


if __name__ == "__main__":
    main()
