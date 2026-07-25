"""
QFDet with Fusion Neck Configuration
"""

_base_ = './qfdet_baseline.py'

# Modify the model to use fusion neck
model = dict(
    type='QFDet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FusionNeck',  # Use fusion neck
        in_channels=256,
        out_channels=256,
        reduction=16,
        num_outs=5
    ),
    # ... rest remains the same
)