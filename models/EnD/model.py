import torch
import torch.nn as nn
from importlib import import_module
import numpy as np


class pattern_norm(nn.Module):
    def __init__(self, scale = 1.0):
        super(pattern_norm, self).__init__()
        self.scale = scale

    def forward(self, input):
        sizes = input.size()
        if len(sizes) > 2:
            input = input.view(-1, np.prod(sizes[1:]))
            input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
            input = input.view(sizes)
        return input


def EnDNet(backbone, n_classes, pretrained = True):
    mod = import_module("models.basemodels")
    cusModel = getattr(mod, backbone)
    model = cusModel(n_classes=n_classes, pretrained=pretrained)
    model.body.avgpool = nn.Sequential(
        model.avgpool,
        pattern_norm()
    )
    return model


def EnDNet3D(backbone, n_classes, pretrained = True):
    
    mod = import_module("models.basemodels_3d")
    cusModel = getattr(mod, backbone)
    model = cusModel(n_classes=n_classes, pretrained=pretrained)
    model.body.avgpool = nn.Sequential(
        model.avgpool,
        pattern_norm()
    )
    return model

def EnDNetMLP(backbone, n_classes, in_features, hidden_features=1024):
    mod = import_module("models.basemodels_mlp")
    cusModel = getattr(mod, backbone)
    model = cusModel(n_classes=n_classes, in_features= in_features, hidden_features=hidden_features)
    model.backbone.fc1 = nn.Sequential(
        model.backbone.fc1,
        pattern_norm()
    )
    return model