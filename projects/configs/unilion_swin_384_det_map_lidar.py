_base_ = './unilion_swin_384_det_map.py'

from projects.configs.lidar_only_utils import make_lidar_only_pipeline
from projects.configs.unilion_swin_384_det_map import (
    data_basic_config,
    test_pipeline as base_test_pipeline,
    train_pipeline as base_train_pipeline,
)


data_root = '/home/renault/repo/dataset/nuscenes/'
train_pipeline = make_lidar_only_pipeline(base_train_pipeline, data_root)
test_pipeline = make_lidar_only_pipeline(base_test_pipeline, data_root)

model = dict(
    fusion=False,
    with_depth_loss=False,
    img_backbone=None,
    img_neck=None,
    img_map2bev=None,
    use_grid_mask=False,
)

eval_config = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
)

# CBGSDataset only builds cfg['dataset']: the pipeline must be set inside it.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            **data_basic_config,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            test_mode=False)),
    val=dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        test_mode=True,
        eval_config=eval_config),
    test=dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        test_mode=True,
        eval_config=eval_config))

load_from = None
resume_from = None
