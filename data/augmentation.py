"""
Data Augmentation Pipeline for GLA-FatigueNet.
Uses Albumentations for efficient image augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(config):
    """Get training data augmentation pipeline."""
    aug_config = config['augmentation']
    img_size = config['data']['image_size']

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=aug_config['horizontal_flip_prob']),
        A.Rotate(limit=aug_config['rotation_limit'], p=0.5, border_mode=0),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config['brightness_limit'],
            contrast_limit=aug_config['contrast_limit'],
            p=0.5
        ),
        A.GaussNoise(
            std_range=(0.02, 0.1),
            p=0.3
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.Sharpen(p=0.5),
        ], p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.CoarseDropout(
            num_holes_range=(1, aug_config['cutout_num_holes']),
            hole_height_range=(16, aug_config['cutout_max_h_size']),
            hole_width_range=(16, aug_config['cutout_max_w_size']),
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(config):
    """Get validation/test data transformation pipeline (no augmentation)."""
    img_size = config['data']['image_size']

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_inference_transforms(config):
    """Get inference-time transforms."""
    return get_val_transforms(config)
