import pretrainedmodels
from .ternausnet import AlbuNet

def get_model(name='se_resnext50_32x4d' ,freeze=False):
    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    mean = model.mean
    std = model.std
    model = AlbuNet(encoder=model)
    model.mean = mean
    model.std = std
    return model
