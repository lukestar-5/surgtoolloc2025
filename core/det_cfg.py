default_scope = 'mmyolo'
_file_client_args = dict(backend='disk')

data_root = ""
dataset_type = "YOLOv5CocoDataset"

metainfo = dict(
    classes=(
        'bipolar dissector',
        'bipolar forceps',
        'cadiere forceps',
        'clip applier',
        'force bipolar',
        'grasping retractor',
        'monopolar curved scissors',
        'needle driver',
        'permanent cautery hook/spatula',
        'prograsp forceps',
        'stapler',
        'suction irrigator',
        'tip-up fenestrated grasper',
        'vessel sealer',
    ),
    palette=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
    ])

train_ann_file = "/data1/miccai2025/train/train_coco.json"
train_data_prefix = ""
train_batch_size_per_gpu = 4
train_num_workers = 10

persistent_workers = True

val_ann_file = "/data1/miccai2025/val/val_coco.json"
val_data_prefix = ""
val_batch_size_per_gpu = 16
val_num_workers = 10

num_classes = 14

base_lr = 0.001
max_epochs = 80
num_epochs_stage2 = 20

img_scale = (864, 864)
img_scales = [
    (640, 640,),
    (320, 320,),
    (960, 960,),
]

random_resize_ratio_range = (0.1, 2.0)
mosaic_max_cached_images = 40
mixup_max_cached_images = 20

deepen_factor = 1.33
widen_factor = 1.25
strides = [8, 16, 32]

norm_cfg = dict(type='BN')
act_cfg = dict(inplace=True, type='SiLU')

lr_start_factor = 1e-05
dsl_topk = 13
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0
weight_decay = 0.05

save_checkpoint_intervals = 10
val_interval_stage2 = 1
max_keep_ckpts = 3

model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)

batch_shapes_cfg = dict(
    batch_size=32,
    extra_pad_ratio=0.5,
    img_size=864,
    size_divisor=32,
    type='BatchShapePolicy')

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=act_cfg),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides,
            widen_factor=widen_factor
        ),
        prior_generator=dict(
            offset=0,
            strides=[8, 16, 32],
            type='mmdet.MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        loss_bbox=dict(loss_weight=2.0, type='mmdet.GIoULoss'),
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes,
            topk=dsl_topk,
            type='BatchDynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    test_cfg=model_test_cfg,
)

train_pipeline_stage2 = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='SingleImageAdaptiveResize', img_size=img_scale[0], size_divisor=32, extra_pad_ratio=0.5),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'pad_param'
        ))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        ann_file=val_ann_file,
        batch_shapes_cfg=batch_shapes_cfg,
        data_prefix=dict(img=val_data_prefix),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = val_dataloader

_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in img_scales
]

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')

launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

resume = False

tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_file_client_args),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]

val_evaluator = dict(
    ann_file=val_ann_file,
    metric='bbox',
    proposal_nums=(100, 1, 10),
    type='mmdet.CocoMetric')

test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=base_lr, type='AdamW', weight_decay=weight_decay),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
)

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=lr_start_factor,
        type='LinearLR'),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2,
        type='mmdet.PipelineSwitchHook'),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)],
    max_epochs=max_epochs,
    val_begin=20,
    val_interval=save_checkpoint_intervals,
)

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
