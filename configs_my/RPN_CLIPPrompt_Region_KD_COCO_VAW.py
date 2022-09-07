checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = None
# custom_hooks = [dict(type='SetSubModelEvalHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

data_root = '/data/kyanchen/prompt/data'
attribute_index_file = dict(
    file=data_root+'/VAW/common2rare_att2id.json',
    att_group='all'
)
model = dict(
    type='RPN_CLIP_Prompter_Region',
    attribute_index_file=attribute_index_file,
    rpn_all=True,  # RPN是否包含属性预测的内容
    need_train_names=[
        # 'img_backbone',
        'img_neck',
        'rpn_head',
        'att_head',
        # 'prompt_learner',
        'logit_scale', 'head',
        'kd_img_align', 'kd_logit_scale',
    ],
    noneed_train_names=[],
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     # load_ckpt_from='../pretrain/faster_rcnn_epoch_12.pth'
    #     init_cfg=dict(type='Pretrained', prefix='backbone.', map_location='cpu',
    #                   checkpoint='results/EXP20220809_3/latest.pth')
    #     # init_cfg=dict(type='Pretrained', prefix='backbone.',
    #     #               checkpoint='../pretrain/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth')
    # ),
    img_backbone=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=False,
        out_indices=[1, 2, 3, 4],
        # backbone_name='ViT-B/16',
        # load_ckpt_from='results/EXP20220903_0/epoch_40.pth',
        precision='fp32',
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # init_cfg=dict(type='Pretrained', prefix='neck.', map_location='cpu',
        #               checkpoint='results/EXP20220809_3/latest.pth'),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        num_convs=3,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    att_head=dict(
        type='ProposalEncoder',
        out_channels=1024,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=28
        ),
        attribute_head=dict(
            type='TransformerAttrHead',
            in_channel=256,
            embed_dim=512,
            num_patches=14*14,
            use_abs_pos_embed=True,
            drop_rate=0.1,
            class_token=True,
            num_encoder_layers=5,
            global_pool=False,
        )
    ),
    # prompt_learner=dict(
    #     type='PromptLearner',
    #     n_ctx=16,
    #     ctx_init='',
    #     c_specific=False,
    #     class_token_position='middle',
    #     load_ckpt_from='../pretrain/t_model.pth'
    # ),
    prompt_learner=dict(
        type='PromptAttributes',
        prompt_config=dict(
            n_prompt=30,
            is_att_specific=False,
            att_position='mid',
            with_att_type=True,
            context_length=77,
            n_prompt_type=None,
            generated_context=False,
            pos_emb=False,
        ),
        load_ckpt_from='results/EXP20220903_0/epoch_40.pth'
    ),
    text_encoder=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=False,
        out_indices=[1, 2, 3, 4],
        # backbone_name='ViT-B/16',
        load_ckpt_from='results/EXP20220903_0/epoch_40.pth',
        precision='fp32',
    ),
    kd_model=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=True,
        out_indices=[],
        load_ckpt_from='results/EXP20220903_0/epoch_40.pth',
        precision='fp32',
    ),
    # text_header=dict(
    #     type='TransformerEncoderHead',
    #     in_dim=1024,
    #     embed_dim=256,
    #     use_abs_pos_embed=False,
    #     drop_rate=0.05,
    #     class_token=False,
    #     num_encoder_layers=1,
    #     global_pool=False,
    # ),
    head=dict(
        type='PromptHead',
        data_root=data_root,
        re_weight_alpha=0.25,
        re_weight_gamma=2,
        re_weight_beta=0.995,
        balance_unk=0.15,
        balance_kd=0.5,
        # kd_model_loss='smooth-l1',
        kd_model_loss='t_ce+ts_ce',
        # balance_kd=1e2,
        # kd_model_loss='ce'
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', class_agnostic=False, iou_threshold=0.7),
            min_bbox_size=4)
    )
)

# dataset settings
dataset_type = 'RPNAttributeDataset'
img_norm_cfg_kd = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=False
# )
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
# img_size = (512, 512)
# img_size = (896, 896)
# img_size = (1024, 1024)
img_size = (1024, 800)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(
        type='Resize',
        img_scale=[(1024, 640), (1024, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', size=img_size, center_pad=True),
    # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.85, 1), prob=0.6),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels', 'dataset_type']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'dataset_type'])
]

kd_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.3]),
    dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg_kd),
    dict(type='Pad', size=(224, 224), center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,  rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size=img_size,  center_pad=True),
            # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.9, 1)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes'])
        ]
    )
]
test_rpn_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,  rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size=img_size,  center_pad=True),
            # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.9, 1)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

# find_unused_parameters = True
samples_per_gpu = 32
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=8,
    # samples_per_gpu=4,
    # workers_per_gpu=0,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='train',
        attribute_index_file=attribute_index_file,
        test_mode=False,
        pipeline=train_pipeline,
        kd_pipeline=kd_pipeline,
        dataset_balance=True
    ),
    val=dict(
        samples_per_gpu=samples_per_gpu,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        attribute_index_file=attribute_index_file,
        test_mode=True,
        pipeline=test_pipeline
    ),
    test=dict(
        samples_per_gpu=12,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        # attribute_index_file=dict(
        #     file=data_root+'/VAW/common2common_att2id.json',
        #     att_group='common1'
        # ),
        attribute_index_file=dict(
            file=data_root + '/VAW/common2rare_att2id.json',
            att_group='all'
        ),
        test_mode=True,
        # pipeline=test_pipeline
        test_rpn=True,
        pipeline=test_rpn_pipeline
    )
)
# #
# optimizer
# optimizer = dict(
#     constructor='SubModelConstructor',
#     sub_model={
#         'prompt_learner': {},
#         # 'text_encoder': {'lr_mult': 0.01},
#         # 'image_encoder': {'lr_mult': 0.01},
#         'neck': {}, 'roi_head': {},
#         'kd_logit_scale': {}, 'kd_img_align': {},
#         'bbox_head': {}, 'logit_scale': {},
#         # 'text_encoder': {'lr_mult': 0.01},
#         # 'kd_model': {'lr_mult': 0.1},
#         # 'kd_logit_scale': {}, 'kd_img_align': {},
#         # 'prompt_learner': {},
#     },
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0005
# )

# # optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model={
        # 'img_backbone': {'lr_mult': 0.01},
        'img_neck': {},
        'rpn_head': {},
        'att_head': {},
        # 'prompt_learner': {},
        'logit_scale': {}, 'head': {},
        'kd_img_align': {}, 'kd_logit_scale': {}
        },
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-3
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[35, 50])

# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=1,
#     warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=60)
evaluation = dict(interval=5, metric='mAP')

load_from = None
# resume_from = 'results/EXP20220905_0/latest.pth'
resume_from = None