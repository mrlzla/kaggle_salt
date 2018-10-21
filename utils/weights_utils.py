import os
import torch

def restore_model(model, weights_path):
    print('Loading the trained models from step {}...'.format(weights_path))
    model.load_state_dict(torch.load(
        weights_path, map_location=lambda storage, loc: storage), strict=False)
    return model

def find_best_weights(weights_path, config=None, mode='max'):
    all_weights = list(map(lambda x: os.path.join(weights_path, x), os.listdir(weights_path)))
    weights = []
    for k in range(config.kfolds):
        cands = [x for x in all_weights if '-{}f-'.format(k) in x]
        mul = -1 if mode == 'max' else 1
        cands = list(sorted(cands, key=lambda x: (mul * float(x.split("-")[3]), -int(x.split('-')[1]))))
        print(cands[:5])
        if len(cands) > 0:
            weights.append(cands[0])
        else:
            weights.append(None)
    #weights = [weights[0], weights[1], weights[4], weights[7], weights[8], weights[9]]
    print(weights)
    
    return weights