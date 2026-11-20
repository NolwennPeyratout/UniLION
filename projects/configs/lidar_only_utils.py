from copy import deepcopy


_IMAGE_ONLY_TRANSFORMS = {'PrepareImageInputs', 'PointToMultiViewDepth'}
_IMAGE_KEYS = {'img', 'img_inputs', 'gt_depth'}


def make_lidar_only_pipeline(pipeline, data_root):
    lidar_pipeline = []
    for pipeline_step in deepcopy(pipeline):
        step_type = pipeline_step['type']
        if step_type in _IMAGE_ONLY_TRANSFORMS:
            continue

        if step_type == 'UnifiedObjectSample':
            pipeline_step['sample_2d'] = False
            db_sampler = pipeline_step.get('db_sampler')
            if db_sampler is not None:
                db_sampler['data_root'] = data_root
                db_sampler['info_path'] = data_root + 'nuscenes_dbinfos_train.pkl'

        if step_type == 'Collect3D':
            pipeline_step['keys'] = [
                key for key in pipeline_step['keys'] if key not in _IMAGE_KEYS
            ]

        if step_type == 'MultiScaleFlipAug3D':
            pipeline_step['transforms'] = make_lidar_only_pipeline(
                pipeline_step['transforms'], data_root
            )

        lidar_pipeline.append(pipeline_step)

    return lidar_pipeline
