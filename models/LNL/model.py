import torch
import torchvision
import torch.nn as nn
from importlib import import_module
from torch.autograd import Function
from torchvision.models.feature_extraction import create_feature_extractor


class LNLGradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
        

def grad_reverseLNL(x):
    return LNLGradReverse.apply(x)


class LNLNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super(LNLNet, self).__init__()
        
        self.backbone = backbone[3:].lower()
        mod = import_module("torchvision.models")
        cusModel = getattr(mod, self.backbone)
        resnet = cusModel(pretrained=pretrained)
        
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.pred_ch = resnet.layer2[-1].conv1.in_channels

        self.returnkey = 'layer2'
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'layer2': self.returnkey, 'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})

    def forward(self, x):
        output = self.body(x)
        return output[self.returnkey_fc], output[self.returnkey]

    def inference(self, x):
        
        output = self.body(x)
        return output[self.returnkey_fc], output[self.returnkey_avg].squeeze()


class LNLPredictor(nn.Module):
    def __init__(self, input_ch, num_classes=2):
        super(LNLPredictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm2d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(input_ch, num_classes) # binary classification, here use sigmoid instead of softmax

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        x2 = self.avgpool(x)
        x2 = x2.view(x2.size(0), -1)
        px = self.linear(x2)
        return x, px


class LNLNet3D(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super(LNLNet3D, self).__init__()
        
        self.backbone = backbone[3:].lower()
        #mod = import_module("torchvision.models.video.r3d_18")
        #cusModel = getattr(mod, self.backbone)
        resnet = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.pred_ch = resnet.layer2[-1].conv1[0].in_channels

        self.returnkey = 'layer2'
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'layer2': self.returnkey, 'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})

    def forward(self, x):
        output = self.body(x)
        return output[self.returnkey_fc], output[self.returnkey]

    def inference(self, x):
        output = self.body(x)
        return output[self.returnkey_fc], output[self.returnkey_avg].squeeze()


class LNLPredictor3D(nn.Module):
    def __init__(self, input_ch, num_classes=2):
        super(LNLPredictor3D, self).__init__()
        self.pred_conv1 = nn.Conv3d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm3d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv3d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(input_ch, num_classes) # binary classification, here use sigmoid instead of softmax

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        x2 = self.avgpool(x)
        x2 = x2.view(x2.size(0), -1)
        px = self.linear(x2)
        return x, px


class LNLNet_MLP(nn.Module):
    def __init__(self, backbone, num_classes, in_features=1024, hidden_features=1024):
        super(LNLNet_MLP, self).__init__()
        
        mod = import_module("models.basemodels_mlp")
        cusModel = getattr(mod, backbone)
        self.net = cusModel(n_classes=num_classes, in_features= in_features, hidden_features=hidden_features)
        hidden_size = hidden_features
        
        self.pred_ch = hidden_size

    def forward(self, x):
        output, hidden = self.net.backbone(x)
        return output, hidden

    def inference(self, x):
        output, hidden = self.net.backbone(x)
        return output, hidden


class LNLPredictor_MLP(nn.Module):
    def __init__(self, input_ch, num_classes=2, hidden_features = 512):
        super(LNLPredictor_MLP, self).__init__()
        self.pred_f1 = nn.Linear(input_ch, hidden_features)
        self.relu = nn.ReLU()
        self.pred_fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.pred_f1(x)
        x = self.relu(x)
        px = self.pred_fc2(x)
        return x, px