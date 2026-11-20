_base_ = './unilion_swin_384_det.py'

data_root = '../dataset/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

model = dict(
    fusion=False,
    with_depth_loss=False,
    img_backbone=None,
    img_neck=None,
    img_map2bev=None,
    use_grid_mask=False,
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='ToEgo'),
    dict(type='LoadMultiTaskAnnotations3D'),
    dict(
        type='VectorizeMap',
        roi_size=(60, 30),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True,
    ),
    dict(
        type='UnifiedObjectSample',
        sample_2d=False,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root=data_root,
            info_path=data_root + 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5,
                ),
            ),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone',
            ],
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2,
            ),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ),
        ),
    ),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(-22.5 * 2, 22.5 * 2),
            scale_lim=(0.9, 1.1),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            tran_lim=[0.5, 0.5, 0.5],
        ),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian',
            'traffic_cone',
        ],
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -3.0, 54.0, 54.0, 5.0],
    ),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -3.0, 54.0, 54.0, 5.0],
    ),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian',
            'traffic_cone',
        ],
    ),
    dict(type='PointShuffle'),
    dict(
        type='MultiTaskFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian',
            'traffic_cone',
        ],
    ),
    dict(
        type='Collect3D',
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_occ',
            'mask_camera',
            'gt_map_labels',
            'gt_map_pts',
            'gt_bev_masks',
            'ego_status',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
        ],
        meta_keys=('filename', 'lidar2img'),
    ),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='ToEgo'),
    dict(type='LoadMultiTaskAnnotations3D'),
    dict(
        type='VectorizeMap',
        roi_size=(60, 30),
        simplify=True,
        permute=False,
        normalize=False,
    ),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(-22.5 * 2, 22.5 * 2),
            scale_lim=(0.9, 1.1),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            tran_lim=[0.5, 0.5, 0.5],
        ),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian',
            'traffic_cone',
        ],
        is_train=False,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type='RandomFlip3D'),
            dict(
                type='MultiTaskFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus',
                    'trailer', 'barrier', 'motorcycle', 'bicycle',
                    'pedestrian', 'traffic_cone',
                ],
                with_label=False,
            ),
            dict(
                type='Collect3D',
                keys=[
                    'points',
                    'gt_bev_masks',
                    'ego_status',
                    'fut_boxes',
                    'gt_ego_fut_trajs',
                    'gt_ego_fut_masks',
                    'gt_ego_fut_cmd',
                ],
                meta_keys=('filename', 'token'),
            ),
        ],
    ),
]

eval_config = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    modality=input_modality,
    pipeline=test_pipeline,
    test_mode=True,
)

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        modality=input_modality,
        pipeline=train_pipeline,
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        modality=input_modality,
        pipeline=test_pipeline,
        eval_config=eval_config,
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        modality=input_modality,
        pipeline=test_pipeline,
        eval_config=eval_config,
    ),
)

load_from = None
resume_from = None
