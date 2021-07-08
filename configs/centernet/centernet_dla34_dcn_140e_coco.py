_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    backbone=dict(type='DLANet', depth=34),
    neck=dict(
        type='CTDLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=80,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100),
    # test_cfg=dict(topK=100,
    #               nms_cfg=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
    #               max_per_img=100)
)

img_norm_cfg = dict(
    mean=[0.408 * 255, 0.447 * 255, 0.470 * 255],
    std=[0.289 * 255, 0.274 * 255, 0.278 * 255],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False,
        test_mode=False,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                crop_size=None,
                ratios=None,
                border=None,
                test_mode=True,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=False,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    # train=dict(pipeline=train_pipeline),
    train=dict(ann_file=data_root + 'annotations/instances_val2017.json',
               img_prefix=data_root + 'val2017/', pipeline=train_pipeline),  # TODO use val because memory issue
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[90, 120])
runner = dict(max_epochs=140)

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=5)
