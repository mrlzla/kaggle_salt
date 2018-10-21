from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import cv2
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import pretrainedmodels.utils as utils

from .albumentations import strong_aug, pad_and_crop, strong_tta

class ToSpaceBGR(object):

    def __init__(self, is_bgr=True):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):

    def __init__(self, is_255=True):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

def pad_img(img, min_height, min_width, border_mode=cv2.BORDER_REFLECT_101):
    #import ipdb; ipdb.set_trace()
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode)

    assert img.shape[0] == max(min_height, height)
    assert img.shape[1] == max(min_width, width)

    return img

class Sampler(data.Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.pad_sizes = [128, 160, 192]
        self.crop_sizes = [96, 128, 160, 192]
        self.pad2_sizes = [224, 256]
        self.crop2_sizes = [160, 192, 224, 256]

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

    def update(self):
        resize_applied = False
        self.resize_applied = resize_applied
        self.data_source.pad_size = np.random.choice(self.pad_sizes) if not resize_applied else np.random.choice(self.pad2_sizes)
        self.data_source.crop_size = np.random.choice(list(filter(lambda x: x <= self.data_source.pad_size, self.crop_sizes))[-3:]) if not resize_applied else \
                                     np.random.choice(list(filter(lambda x: x <= self.data_source.pad_size, self.crop2_sizes))[-3:])

class BatchSampler(data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last=True):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        self.sampler.update()
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class Dataset(data.Dataset):
    def __init__(self, datadir, config, phase='train', fold_idx=None, mean=None, std=None, use_tta=False):
        """Initialize and preprocess the CelebA dataset."""
        if phase not in ['train', 'val', 'test']:
            raise(ValueError())
        self.phase=phase
        self.datadir = datadir
        self.config = config
        self.train_csv = os.path.join(datadir, 'train.csv')
        self.depths_csv = os.path.join(datadir, 'depths.csv')
        self.images_path = os.path.join(self.datadir,'images')
        self.masks_path = os.path.join(self.datadir,'masks')
        self.h, self.w = config.crop_size
        self.new_h = int(np.ceil(self.h / config.output_stride) * config.output_stride)
        self.new_w = int(np.ceil(self.w / config.output_stride) * config.output_stride)
        self.size = (self.new_h, self.new_w)
        self.pad_size=192
        self.crop_size=192
        self.up_size=224
        self.use_tta=use_tta
        self.use_resize=config.use_resize
        self.resize_applied = False if not self.use_tta else True

        self.albumentations = strong_aug()
        self.tta = strong_tta()
        self.transform = T.Compose([
            T.ToTensor(),
            ToSpaceBGR(),
            ToRange255(),
            T.Normalize(mean=mean, std=std)
        ])
        self.preprocess(config, fold_idx)

    def preprocess(self, config, fold_idx):
        self.df = pd.read_csv(self.train_csv, index_col="id", usecols=[0])
        self.depths_df = pd.read_csv(self.depths_csv, index_col="id")
        if self.phase in ['train', 'val']:
            self.df = self.df.join(self.depths_df)
            self.df['masks'] = [np.clip(imread(os.path.join(self.masks_path, filename + '.png'), as_gray=True), 0, 255)  for filename in tqdm(self.df.index)]
            self.df["coverage"] = self.df.masks.map(lambda x: x / 255.0).map(np.sum)
            #print(self.df.coverage.max())
            #self.df.masks[self.df.coverage == 0].map(lambda x: 255*np.ones([101, 101])) 
            self.df['images'] = [imread(os.path.join(self.images_path, filename + '.png'))[:, :, :3] for filename in tqdm(self.df.index)]
            self.df['coverage_images'] = self.df.images.map(lambda x: x / 255.0).map(np.sum)
            self.df = self.df[self.df.coverage_images > 0]
            def cov_to_class(val):    
                for i in range(0, 11):
                    if val / self.h / self.w * 10 <= i :
                        return i
            self.df["coverage_class"] = self.df.coverage.map(cov_to_class)
            self.df.sort_values(by=["coverage_class", "z"])
            #self.df = self.df.sample(frac=1, random_state=12)
            self.folds = [[] for _ in range(config.kfolds)]
            for i in range(0, 11):
                class_items = self.df.index[self.df.coverage_class == i]  
                for j, item in enumerate(class_items):
                    self.folds[j % config.kfolds].append(item)
            self.data = self.folds[fold_idx] if self.phase == 'val' else sum(self.folds[:fold_idx] + self.folds[fold_idx+1:], [])
        else:
            self.df = self.depths_df[~self.depths_df.index.isin(self.df.index)]
            self.df['images'] = [imread(os.path.join(self.images_path, filename + '.png'))[:, :, :3] for filename in tqdm(self.df.index)]
            self.data = self.df.index
        self.num_samples = len(self.data)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = self.df.images[self.data[index]].copy()
        zero_flag = int(img.max() < 5)
        if zero_flag and self.phase == 'test':
            return [img, 0, self.data[index], zero_flag]
        if self.phase in ['train', 'val']:
            mask = self.df.masks[self.data[index]].copy()
            if self.use_resize or self.use_tta:
                if self.resize_applied:
                    img, mask = cv2.resize(img, None, fx=2, fy=2, interpolation= cv2.INTER_NEAREST), \
                                cv2.resize(mask, None, fx=2, fy=2, interpolation= cv2.INTER_NEAREST)
            inp_dict = {'image': img, 'mask': mask}
            if self.phase=='train':
                inp_dict = self.albumentations(**inp_dict)
                res = pad_and_crop(self.pad_size, self.crop_size, self.resize_applied)(**inp_dict)
            else:
                res = pad_and_crop(self.size[0] if not self.resize_applied else self.up_size)(**inp_dict)

            img, mask = res['image'], res['mask']
            mask = np.expand_dims(mask, 2).astype(np.uint8)
            if self.config.use_depth:
                depth_val = self.depths_df.z[self.data[index]]
                mean = 506.453
                std = 959.0
                depth = np.zeros_like(mask).astype(np.float32)
                depth += (depth_val - mean) / std
                return self.transform(img), T.ToTensor()(depth), T.ToTensor()(mask)
            else:
                return self.transform(img), 0, T.ToTensor()(mask)
        else:
            if self.use_tta:
                img = cv2.resize(img, None, fx=2, fy=2, interpolation= cv2.INTER_NEAREST)
                inp_dict = {'image' : img}
                img = pad_and_crop(self.up_size)(**inp_dict)['image']
            else:
                inp_dict = {'image' : img}
                img = pad_and_crop(128)(**inp_dict)['image']
            if self.config.use_depth:
                depth_val = self.depths_df.z[self.data[index]]
                mean = 506.453
                std = 959.0
                depth = np.zeros_like(img)[..., :1].astype(np.float32)
                depth += (depth_val - mean) / std
                return [self.transform(img), T.ToTensor()(depth), self.data[index], zero_flag]
            else:
                return [self.transform(img), 0, self.data[index], zero_flag]
