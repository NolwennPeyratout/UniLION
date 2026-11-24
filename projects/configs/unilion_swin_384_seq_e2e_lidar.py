_base_ = './unilion_swin_384_seq_e2e.py'

from projects.configs.lidar_only_utils import make_lidar_only_pipeline
from projects.configs.unilion_swin_384_seq_e2e import (
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

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        eval_config=eval_config,
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        eval_config=eval_config,
    ),
)

#load_from = (
#    'work_dirs/unilion_swin_384_seq_perception_lidar/iter_42192.pth'
#)

load_from = (
    '/home/renault/repo/UniLION/work_dirs/unilion_swin_384_det_map_lidar/epoch_1.pth'
)
resume_from = None
