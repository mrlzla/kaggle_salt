import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import cv2
import tqdm
import numpy as np
import itertools
from torchvision import transforms as T
from torch.utils.data import DataLoader
from nnet.model_builder import get_model
from nnet.callbacks import update_lr, save_checkpoint
from nnet.losses import get_loss
from nnet.metrics import get_metric
from dataset.albumentations import strong_tta, pad_and_crop
from dataset.data_loader import Dataset, Sampler, BatchSampler
from torch.backends import cudnn
from config import config
from utils.weights_utils import find_best_weights, restore_model
from utils.rle import rle_encoding, mask_to_rles
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes
from skimage.morphology import label

def crop(img, h, w):
    new_h, new_w = img.shape
    top_ind = int((new_h - h) / 2.0)
    bottom_ind = new_h - h - top_ind
    left_ind = int((new_w - w) / 2.0)
    right_ind = new_w - w - left_ind
    return img[top_ind:-bottom_ind, left_ind:-right_ind]

def test(args, cfg):
    cudnn.benchmark = True
    device = torch.device(
            'cuda' if args.gpu=="0" else 'cpu')
    if args.weights:
        if os.path.isdir(args.weights):
            weights = find_best_weights(args.weights, cfg)
            weights = [weights[0], weights[1]]
            print(weights)
            models = [restore_model(get_model(), weight) for weight in weights]
        else:
            models = [restore_model(get_model(), args.weights)]
    else:
        models = [get_model(freeze=True) for _ in range(cfg.kfolds)]
    for model in models:
        model.eval()
    with open('csv/res_{}.csv'.format(args.weights.split('/')[1]), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'rle_mask'])
        test_set = Dataset(args.datadir, config, phase='test', fold_idx=None, mean=model.mean, std=model.std)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        tta = pad_and_crop(224)
                
        for image, depth, image_id, zero_flag in tqdm.tqdm(test_loader):
            if zero_flag.numpy()[0]:
                mask = np.zeros([cfg.h, cfg.w, 1], dtype=np.uint8)
            else:
                image = image.to(device)
                depth = depth.to(device)
                image = torch.cat([image, image.flip(3)], dim=0)
                depth = torch.cat([depth, depth], dim=0)
                logits = torch.zeros(128, 128, dtype=torch.float32).to(device)
                for model in models:
                    model.to(device)
                    y_pred = model(image, depth)
                    logits += torch.cat([y_pred[0].unsqueeze(0), y_pred[1].unsqueeze(0).flip(3)], dim=0).mean(dim=0).squeeze(0)
                mask = ((logits / len(models))>0.5).byte().cpu().numpy()
                if cfg.use_tta:
                    mask = pad_and_crop(224, 202)(image=mask)['image']
                    mask = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                else:
                    mask = crop(mask, cfg.h, cfg.w)[:, :, np.newaxis]
            mask = remove_small_objects(label(mask), 10).astype(np.uint8)
            mask[mask > 1] = 1
            rles = list(mask_to_rles(mask))
            for rle in rles:
                writer.writerow(
                    [image_id[0], ' '.join(str(y) for y in rle)])
            if len(rles) == 0:
                writer.writerow([image_id[0], ''])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        default='data')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--gpu', default="0")
    args = parser.parse_args()
    config = config()
    test(args, config)
