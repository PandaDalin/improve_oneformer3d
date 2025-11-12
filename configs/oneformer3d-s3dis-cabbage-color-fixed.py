_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 64
num_instance_classes = 3
num_semantic_classes = 3

model = dict(
    type='S3DISCabbageOneFormer3D',  # 使用甘蓝专用模型
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.028,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True
        ),
    decoder=dict(
        type='EnhancedQueryDecoder',  # 使用增强的解码器
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=400,  
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,  
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
        enable_cabbage_refinement=True,  # 启用甘蓝球径增强
        enable_leaf_instance_enhancement=False,  # 启用叶片实例增强
        enable_integrated_enhancement=False,  # 启用综合增强模块
        refinement_config=dict(
            in_channels=num_channels,
            hidden_channels=128,
            num_classes=num_instance_classes,
            refinement_layers=3,
            attention_heads=8,
            dropout=0.1
        ),
        # 叶片实例增强配置
        leaf_instance_config=dict(
            openformer_feature_dim=256,
            leaf_feature_dim=192,
            embedding_dim=96,
            k=12,  # 邻居点数
            enable_separation=True,  # 启用叶片分离
            enable_completion=True,  # 启用叶片补全
            enable_boundary_detection=True,  # 启用边界检测
            enable_instance_consistency=True  # 启用实例一致性
        ),
        # 综合增强模块配置
        integrated_enhancement_config=dict(
            openformer_feature_dim=256,
            enable_cross_module_interaction=True,  # 启用跨模块交互
            enable_output_integration=True,  # 启用输出整合
            enable_quality_assessment=True,  # 启用质量评估
            cross_interaction_config=dict(
                feature_dim=256,
                attention_heads=8,
                dropout=0.1
            ),
            output_integration_config=dict(
                feature_dim=256,
                fusion_layers=2,
                dropout=0.1
            )
    ),
    criterion=dict(
        type='S3DISEnhancedUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        inst_criterion=dict(
            type='InstanceCriterion',
            num_classes=num_instance_classes,
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                ]
            ),
            non_object_weight=0.05,
            iter_matcher=True,
            fix_dice_loss_weight=True,
            fix_mean_loss=True
        ),
        sem_criterion=dict(
            type='S3DISCabbageFocalCriterion',
            loss_weight=5.0,
            focal_loss=dict(
                type='CabbageSoftmaxFocalLoss',
                num_classes=num_semantic_classes,
                loss_weight=1.0,
                gamma=2.0,
                class_weight=[6.0, 1.1, 1.0],  # 甘蓝头、地面、叶子的权重
                reduction='mean',
                ignore_index=255
            )
        ),
        enable_cabbage_refinement=True,
        enable_leaf_instance_loss=False,
        enable_integrated_loss=False,
        cabbage_refinement_config=dict(
            loss_weight=1.0,
            refinement_weight=0.3,
            shape_weight=0.5,
            size_weight=0.8,
            smoothness_weight=0.3,
            boundary_weight=2.0,
            connectivity_weight=0.6,
            consistency_weight=0.2
        )
    ),
    test_cfg=dict(
        inst_score_thr=0.0,
        matrix_nms_kernel='linear',
        nms=True,
        npoint_thr=100,
        num_sem_cls=num_semantic_classes,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        pan_score_thr=0.4,
        sp_score_thr=0.15,
        stuff_cls=[0, 1],  # corm, ground
        thing_cls=[2],     # leaf
        topk_insts=450
    ),
    train_cfg=dict()
)

# dataset settings
dataset_type = 'S3DISSegDataset_'
data_root = 'data/s3dis/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask'
)

# 甘蓝专用类别映射
class_names = ['corm', 'ground', 'leaf']
label2cat = {0: 'corm', 1: 'ground', 2: 'leaf'}
sem_mapping = [0, 1, 2]
inst_mapping = [0, 1, 2]

# 数据加载配置 - 修复颜色处理
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample_',
        num_points=120000),  
    dict(type='PointInstClassMapping_',
         num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        shift_height=False,
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),  # 甘蓝数据集的颜色均值
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5]),  # 甘蓝数据集的颜色均值
        ]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

# 数据集配置
train_dataloader = dict(
    batch_size=2,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='s3dis_infos_Area_5.pkl',
                data_prefix=data_prefix,
                pipeline=train_pipeline,
                filter_empty_gt=True,
                box_type_3d='Depth')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='s3dis_infos_Area_2.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True,
        box_type_3d='Depth')
)

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='UnifiedSegMetric',
    metric_meta=dict(
        dataset_name='S3DIS',
        classes=class_names,
        label2cat=label2cat,
        ignore_index=[]
    ),
    inst_mapping=inst_mapping,
    sem_mapping=sem_mapping,
    stuff_class_inds=[0, 1],  # corm, ground
    thing_class_inds=[2],     # leaf
    min_num_points=1,
    id_offset=65536
)

test_evaluator = val_evaluator

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=512, val_interval=8)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# 学习率调度器
param_scheduler = dict(
    type='PolyLR',
    begin=0,
    end=512,
    power=0.9
)

# 默认钩子
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=16,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# 其他配置
load_from = None
resume = False
work_dir = 'work_dirs/cabbage_color_fixed'
launcher = 'none'
