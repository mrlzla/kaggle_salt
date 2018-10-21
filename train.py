import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import itertools
import gc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from nnet.model_builder import get_model
from nnet.callbacks import update_lr, save_checkpoint
from nnet.losses import get_loss
from nnet.metrics import get_metric
from dataset.data_loader import Dataset, Sampler, BatchSampler
from torch.backends import cudnn
from config import config
from utils.weights_utils import find_best_weights, restore_model

def train(args, cfg):
    #cudnn.benchmark = True
    device = torch.device(
            'cuda' if args.gpu=="0" else 'cpu')
    if args.weights:
        #import ipdb; ipdb.set_trace()
        weights = find_best_weights(args.weights, cfg)
        #end = cfg.start_fold if cfg.start_cycle == 0 else cfg.start_fold + 1
        models = [restore_model(get_model(), weight) if weight else get_model(freeze=True) for weight in weights]
    else:
        models = [get_model(freeze=True) for _ in range(cfg.kfolds)]
    lr = cfg.lr
    for fold in range(cfg.start_fold, cfg.kfolds):
        #if fold in [0, 1, 2]:
        #    continue
        best_metric = float(weights[fold].split('-')[3]) if args.weights and weights[fold]  else -np.inf
        best_tta_metric = -np.inf
        start_cycle = cfg.start_cycle if fold == cfg.start_fold else 0
        
        for cycle in range(start_cycle, cfg.num_cycles):
            #import ipdb; ipdb.set_trace()
            lr = cfg.start_lr if fold == cfg.start_fold and cycle==cfg.start_cycle else lr
            alpha_zero = lr

            if cycle != 0 and best_metric < 0.83:
                break
            model = models[fold]
            print(sum(p.numel() for p in model.parameters()))
            print("Best metric is {}".format(best_metric))
            optimizer = torch.optim.Adam(model.parameters(), lr = lr) if cycle == 0 else torch.optim.RMSprop(model.parameters(), lr = lr)
            lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=cfg.lr_mult, patience=16, verbose=True)
            metric_fn = get_metric()
            model.to(device)
            train_set = Dataset(args.datadir, config, phase='train', fold_idx=fold, mean=model.mean, std=model.std)
            val_set = Dataset(args.datadir, config, phase='val', fold_idx=fold, mean=model.mean, std=model.std)
            val_set_tta = Dataset(args.datadir, config, phase='val', fold_idx=fold, mean=model.mean, std=model.std, use_tta=True)
            train_sampler = BatchSampler(Sampler(train_set), cfg.batch_size)
            train_loader = DataLoader(train_set,  batch_sampler=train_sampler, num_workers=8)
            val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)
            val_tta_loader = DataLoader(val_set_tta, batch_size=cfg.batch_size, shuffle=True)


            def train_step(loader):
                train_loss = []
                train_metric = []
                for image, depth, mask in tqdm.tqdm(loader):
                    image = image.type(torch.float).to(device)
                    mask = mask.to(device)
                    depth = depth.to(device)
                    y_pred = model(image, depth)
                    loss = loss_fn(y_pred, mask)

                    metric = metric_fn(y_pred, mask)
                    train_metric.append(metric.item())

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    train_loss.append(loss.item())
                return train_loss, train_metric

            def val_step(loader):
                val_loss = []
                val_metric = []
                with torch.no_grad():
                    for image, depth, mask in tqdm.tqdm(loader):
                        image = image.to(device)
                        mask = mask.to(device)
                        depth = depth.to(device)
                        y_pred = model(image, depth)

                        loss = loss_fn(y_pred, mask)
                        metric = metric_fn((y_pred>0.5).byte(), mask.byte())
                        val_loss.append(loss.item())
                        val_metric.append(metric.item())
                return val_loss, val_metric

            if cycle == 0 and not args.weights:
                loss_fn = get_loss(0, freeze_epochs=True)

                for epoch in range(cfg.frozen_epochs):
                    print(train_set.pad_size, " ", train_set.crop_size)
                    
                    train_loss, train_metric = train_step(train_loader)
                    
                    print("Epoch: %d, Train Loss: %.4f, Train Metric: %.4f," % (epoch, 
                                                                               np.mean(train_loss), 
                                                                               np.mean(train_metric)))

            for param in model.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            
            num_epochs = cfg.num_epochs if cycle > 0 else cfg.start_epochs
            start_epoch = cfg.start_epoch if cycle==start_cycle and fold==cfg.start_fold else 0
            for epoch in range(start_epoch, num_epochs):
                loss_fn = get_loss(cycle)
                print(loss_fn)
                print(train_set.pad_size, " ", train_set.crop_size)
                
                train_loss, train_metric = train_step(train_loader)
                
                val_loss, val_metric = val_step(val_loader)

                val_tta_loss, val_tta_metric = val_step(val_tta_loader)

                print("Epoch: %d, Train Loss: %.4f, Train Metric: %.4f, Val Loss: %.4f, Val Metric: %.4f, Val TTA Loss: %.4f, Val TTA Metric: %.4f" % (epoch, 
                                                                                                               np.mean(train_loss), np.mean(train_metric),
                                                                                                               np.mean(val_loss), np.mean(val_metric),
                                                                                                               np.mean(val_tta_loss), np.mean(val_tta_metric),
                                                                                                               ))
                lr = update_lr(optimizer, cfg, lr, epoch, alpha_zero=alpha_zero, verbose=1)
                
                lr_scheduler.step(np.mean(val_metric))
                new_lr = optimizer.param_groups[0]['lr']
                if abs(new_lr - lr) > 1e-7:
                    alpha_zero = max(lr*cfg.lr_mult, 1.6e-5)
                best_metric, best_tta_metric = save_checkpoint(model, np.mean(val_metric), best_metric, np.mean(val_tta_metric), best_tta_metric, fold, epoch, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        default='data')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--gpu', default="0")
    args = parser.parse_args()
    config = config()
    train(args, config)
