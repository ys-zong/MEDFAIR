import torch
import torchvision
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


class cusResNet18_3d(nn.Module):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet18_3d, self).__init__()
        resnet = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
        self.avgpool = resnet.avgpool
        
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})

    def forward(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()

    
class cusResNet50_3d(cusResNet18_3d):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet50_3d, self).__init__(n_classes, pretrained)
        resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        
        resnet.blocks[-1].proj = nn.Linear(2048, n_classes)
        self.avgpool = resnet.blocks[-1].pool
        
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'blocks.5.pool': self.returnkey_avg, 'blocks.5.proj': self.returnkey_fc})
    
    
class MLPclassifer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPclassifer, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        x = self.relu(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x    