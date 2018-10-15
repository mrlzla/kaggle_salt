from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, RandomGamma, RandomSizedCrop,
    Flip, OneOf, Compose, ElasticTransform, IAAPerspective, HorizontalFlip, PadIfNeeded, RandomCrop
)

import numpy as np

def strong_tta(p=0.9):
    return Compose([
        Blur(blur_limit=3),
        OneOf([
            RandomContrast(p=.5),
            RandomBrightness(p=.5)
        ]),
        RandomGamma(),
        #RandomSizedCrop(p=0.9, min_max_height=(65, 101), height=101, width=101),
    ], p=p)

def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        Blur(blur_limit=3),
        GridDistortion(p=.2),
        #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=0.0, p=0.7),
        # OneOf([
        #     GridDistortion(p=.2),
        #     ElasticTransform(p=.2)
        # ], p=0.2),
        RandomContrast(p=.5),
        RandomBrightness(p=.5),
        RandomGamma(),
        #RandomSizedCrop(p=0.9, min_max_height=(65, 101), height=101, width=101),
    ], p=p)

def pad_and_crop(pad_size, crop_size=None, resize_applied=False):
    if crop_size:
        return Compose([
            PadIfNeeded(pad_size, pad_size),
            RandomCrop(crop_size, crop_size)
            
        ], p=1.0)
    else:
        return PadIfNeeded(pad_size, pad_size)


def aug_mega_hardcore(p=.95):
    return Compose([
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(p=.25),
            IAAEmboss(p=.25)
        ], p=.35),
        OneOf([
            IAAAdditiveGaussianNoise(p=.3),
            GaussNoise(p=.7),
        ], p=.5),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.3),
            Blur(blur_limit=3, p=.5),
        ], p=.4),
        OneOf([
            RandomContrast(p=.5),
            RandomBrightness(p=.5),
        ], p=.4),
        ShiftScaleRotate(shift_limit=.0, scale_limit=.45, rotate_limit=45, p=.7),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.2),
            ElasticTransform(p=.2),
            IAAPerspective(p=.2),
            IAAPiecewiseAffine(p=.3),
        ], p=.6),
        HueSaturationValue(p=.5)
    ], p=p)
