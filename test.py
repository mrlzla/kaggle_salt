import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import tqdm
import numpy as np
import itertools
from torchvision import transforms as T
from torch.utils.data import DataLoader
from nnet.model_builder import get_model
from nnet.callbacks import update_lr, save_checkpoint
from nnet.losses import get_loss
from nnet.metrics import get_metric
from dataset.albumentations import strong_tta
from dataset.data_loader import Dataset, Sampler, BatchSampler
from torch.backends import cudnn
from config import config
from utils.weights_utils import find_best_weights, restore_model
from utils.rle import rle_encoding, mask_to_rles

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
            weights = find_best_weights(args.weights.split('/')[0], cfg)
            models = [restore_model(get_model(), weight) for weight in weights]
            models += [get_model(freeze=True) for _ in range(cfg.kfolds - len(models))]
        else:
            models = [restore_model(get_model(), args.weights)]
    else:
        models = [get_model(freeze=True) for _ in range(cfg.kfolds)]
    with open('csv/res_{}.csv'.format(args.weights.split('/')[1]), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'rle_mask'])
        fold=0
        model = models[fold]
        model.eval()
        model.to(device)
        test_set = Dataset(args.datadir, config, phase='test', fold_idx=fold, mean=model.mean, std=model.std)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        tta = strong_tta()
                
        for image, depth, image_id, zero_flag in tqdm.tqdm(test_loader):
            if zero_flag.numpy()[0]:
                mask = np.zeros([cfg.h, cfg.w, 1], dtype=np.uint8)
            else:
                image = image.to(device)
                if cfg.use_tta:
                    inp_dict = {'image': image}
                    inp_dict = tta(**inp_dict)
                    image = inp_dict['image']
                depth = depth.to(device)
                image = torch.cat([image, image.flip(3)], dim=0)
                depth = torch.cat([depth, depth], dim=0)
                y_pred = model(image, depth)
                logits = torch.cat([y_pred[0].unsqueeze(0), y_pred[1].unsqueeze(0).flip(3)], dim=0).mean(dim=0)
                mask = (logits>0.5).byte().squeeze(0).cpu().numpy()
                mask = crop(mask, cfg.h, cfg.w)[:, :, np.newaxis]
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
