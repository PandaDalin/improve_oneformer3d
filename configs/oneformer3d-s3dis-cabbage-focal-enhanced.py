_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 64
num_instance_classes = 3
num_semantic_classes = 3

model = dict(
    type='S3DISCabbageOneFormer3D',  # 使用增强的甘蓝专用模型
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
        enable_leaf_instance_enhancement=True,  # 启用叶片实例增强
        enable_integrated_enhancement=True,  # 启用综合增强模块
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
            leaf_feature_dim=128,
            embedding_dim=64,
            k=16,  # 邻居点数
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
        )),
    criterion=dict(
        type='S3DISEnhancedUnifiedCriterion',  # 使用增强的损失函数
        num_semantic_classes=num_semantic_classes,
        enable_cabbage_refinement=False,  # 启用甘蓝球径增强损失
        enable_leaf_instance_loss=True,  # 启用叶片实例损失
        enable_integrated_loss=False,  # 启用综合损失
        cabbage_refinement_config=dict(
            refinement_weight=0.3,
            consistency_weight=0.2,
            boundary_weight=2.0,
            shape_weight=0.5,
            smoothness_weight=0.3,
            size_weight=0.8,
            connectivity_weight=0.6,
            loss_weight=1.0
        ),
        # 叶片实例损失配置
        leaf_instance_loss_config=dict(
            seg_weight=1.0,
            head_weight=0.8,
            leaf_weight=0.6,
            instance_weight=0.4,
            quality_weight=0.2,
            margin=0.5,  # 实例间分离边界
            enable_intra_loss=True,  # 启用实例内损失
            enable_inter_loss=True,  # 启用实例间损失
            enable_boundary_loss=True,  # 启用边界损失
            enable_completion_loss=True  # 启用补全损失
        ),
        # 综合损失配置
        integrated_loss_config=dict(
            seg_weight=1.0,
            head_weight=0.8,
            leaf_weight=0.6,
            instance_weight=0.4,
            quality_weight=0.2,
            consistency_weight=0.3,
            cross_module_weight=0.4
        ),
        sem_criterion=dict(
            type='S3DISCabbageFocalCriterion',  # 使用甘蓝专用Focal Loss
            loss_weight=5.0,
            focal_loss=dict(
                type='CabbageSoftmaxFocalLoss',
                num_classes=3,
                gamma=2.0,
                class_weight=[6.0, 1.1, 1.0],  # [corm, ground, leaf] - 增加球径权重
                ignore_index=255,
                reduction='mean',
                loss_weight=1.0)),  
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],  
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    # 甘蓝专用配置
    cabbage_config=dict(
        enable_cabbage_refinement=True,
        enable_stage_classification=True,
        enable_leaf_instance_enhancement=True,  # 启用叶片实例增强
        enable_integrated_enhancement=True,  # 启用综合增强
        refinement_config=dict(
            in_channels=num_channels,
            hidden_channels=128,
            num_classes=num_instance_classes,
            refinement_layers=3,
            attention_heads=8,
            dropout=0.1
        ),
        stage_classifier_config=dict(
            in_channels=num_channels,
            hidden_channels=128,
            num_stages=3,  # 幼苗期、生长期、结球期
            dropout=0.1
        ),
        # 叶片实例增强配置
        leaf_instance_config=dict(
            openformer_feature_dim=256,
            leaf_feature_dim=128,
            embedding_dim=64,
            k=16,
            enable_separation=True,
            enable_completion=True,
            enable_boundary_detection=True,
            enable_instance_consistency=True,
            separation_config=dict(
                feature_dim=128,
                k=16,
                threshold=0.5
            ),
            completion_config=dict(
                feature_dim=128,
                k=16,
                gap_radius=0.05
            ),
            boundary_config=dict(
                feature_dim=128,
                k=16,
                threshold=0.7
            ),
            consistency_config=dict(
                embedding_dim=64,
                similarity_threshold=0.7,
                neighbor_radius=0.03
            )
        ),
        # 综合增强配置
        integrated_enhancement_config=dict(
            openformer_feature_dim=256,
            enable_cross_module_interaction=True,
            enable_output_integration=True,
            enable_quality_assessment=True,
            cross_interaction_config=dict(
                feature_dim=256,
                attention_heads=8,
                dropout=0.1,
                interaction_layers=2
            ),
            output_integration_config=dict(
                feature_dim=256,
                fusion_layers=2,
                dropout=0.1,
                smoothing_factor=0.7
            ),
            quality_assessment_config=dict(
                enable_head_quality=True,
                enable_leaf_quality=True,
                enable_consistency_quality=True,
                quality_weights=[0.4, 0.4, 0.2]
            )
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=450,  
        inst_score_thr=0.0, 
        pan_score_thr=0.4, 
        npoint_thr=100,  
        obj_normalization=True,
        obj_normalization_thr=0.01,  
        sp_score_thr=0.15,  
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0, 1],
        thing_cls=[2]))

# dataset settings
dataset_type = 'S3DISSegDataset_'
data_root = '/home/lihongda/3D/oneformer3d/mmdetection3d/data/s3dis/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_area = [5]
test_area = 2

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
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],  
        scale_ratio_range=[0.9, 1.1],  
        translation_std=[.1, .1, .1],  
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
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
        with_bbox_3d=False,
        with_label_3d=False,
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
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

# run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
            type='ConcatDataset',
            datasets=([
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f's3dis_infos_Area_{i}.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    data_prefix=data_prefix,
                    box_type_3d='Depth',
                    backend_args=None) for i in train_area])))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='Depth',
        backend_args=None))
test_dataloader = val_dataloader

class_names = [
    'corm', 'ground', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],  
    classes=class_names,
    dataset_name='S3DIS')
sem_mapping = [0, 1, 2]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0,1],
    thing_class_inds=[2],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),  
    clip_grad=dict(max_norm=10, norm_type=2))  
param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9)  

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=16,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

load_from = None #'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=512, val_interval=8)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
