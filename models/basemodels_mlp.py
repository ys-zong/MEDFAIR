import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=1024, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        

    def forward(self, x):
        x1 = self.fc1(x)
        x_hidden = self.relu(x1)
        x_out = self.fc2(x_hidden)
        return x_out, x_hidden.squeeze()


class cusMLP(nn.Module):    
    def __init__(self, n_classes, in_features, hidden_features, disentangle = False):
        super(cusMLP, self).__init__()
        self.backbone = MLP(in_features, hidden_features, n_classes)
        
        if disentangle is True:
            self.backbone.fc2 = nn.Linear(hidden_features * 2, n_classes)

    def forward(self, x):
        outputs, hidden = self.backbone(x)
        return outputs, hidden

    def inference(self, x):
        outputs, hidden = self.backbone(x)
        return outputs, hidden
   

class MLPclassifer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPclassifer, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        x = self.relu(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x    