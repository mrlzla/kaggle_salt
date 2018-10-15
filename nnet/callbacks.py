import torch
import numpy as np
from utils.telegram import send_message

def _cosine_anneal_schedule(t, T, alpha_zero=1e-2):
    M=10
    cos_inner = np.pi * (t % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return max(float(alpha_zero / 2 * cos_out), alpha_zero / 10)

def update_lr(opt, cfg, lr, epoch, alpha_zero=None, verbose=0):
    lr = _cosine_anneal_schedule(epoch+1, 80, alpha_zero=alpha_zero)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    if verbose:
        print("Lr is equal to {}".format(lr))
    return lr

def save_checkpoint(model, val_metric, best_metric, val_tta_metric, best_tta_metric, fold, epoch, verbose=0):
    if val_metric > best_metric or val_tta_metric > best_tta_metric:
        best_metric = val_metric if val_metric > best_metric else best_metric
        best_tta_metric = val_tta_metric if val_tta_metric > best_tta_metric else best_tta_metric
        path = "weights/weights-{}-{}f-{}-salt.pth".format(epoch, fold, best_metric)
        torch.save(model.state_dict(), path)
        if verbose:
            print("model saved to {}".format(path))
            send_message("model saved to {}".format(path))
    return best_metric, best_tta_metric



