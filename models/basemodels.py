import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class cusResNet18(nn.Module):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
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
    
    
class cusResNet50(cusResNet18):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet50, self).__init__(n_classes, pretrained)
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)

        self.avgpool = resnet.avgpool
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})
    

class cusDenseNet121(cusResNet18):    
    def __init__(self, n_classes, pretrained = True, disentangle = False):
        super(cusDenseNet121, self).__init__(n_classes, pretrained)
        resnet = torchvision.models.densenet121(pretrained=pretrained)
        
        resnet.classifier = nn.Linear(resnet.classifier.in_features, n_classes)
       
        self.returnkey_fc = 'classifier'
        self.body = create_feature_extractor(
            resnet, return_nodes={'classifier': self.returnkey_fc})
    
    def forward(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_fc]

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_fc]


class MLPclassifer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPclassifer, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self,x):
        x = self.relu(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x    