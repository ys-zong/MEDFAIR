import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from torch.autograd import Function


class GradReverse(Function):
    """
    Implement the gradient reversal layer adapting from domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class CFairNet(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, pretrained = True):
        super(CFairNet, self).__init__()
        
        self.num_classes = num_classes
        self.used_classes = 2 # can only handle binary attributes
        mod = import_module("models.basemodels")
        cusModel = getattr(mod, backbone)
        self.net = cusModel(n_classes=self.num_classes, pretrained=pretrained)
        hidden_size = self.net.body.fc.in_features
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [hidden_size] + [adversary_size]
        self.num_adversaries_layers = len([adversary_size])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.used_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.used_classes)])
        
    def forward(self, inputs, labels):
        h_relu = inputs
        outputs, features = self.net(h_relu)
        h_relu = F.relu(features)
        
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        
        for j in range(self.used_classes):
            idx = labels[:, 0] == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        return outputs, c_losses

    def inference(self, inputs):
        outputs, features = self.net(inputs)
        return outputs, features

class CFairNet3D(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, pretrained = True):
        super(CFairNet3D, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.used_classes = 2 # can only handle binary attributes
        mod = import_module("models.basemodels_3d")
        cusModel = getattr(mod, self.backbone)
        self.net = cusModel(n_classes=self.num_classes, pretrained=pretrained)
        hidden_size = self.net.body.fc.in_features
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [hidden_size] + [adversary_size]
        self.num_adversaries_layers = len([adversary_size])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.used_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.used_classes)])

    def forward(self, inputs, labels):
        h_relu = inputs
        outputs, features = self.net(h_relu)
        h_relu = F.relu(features)
        
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        
        for j in range(self.used_classes):
            idx = labels[:, 0] == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        return outputs, c_losses

    def inference(self, inputs):
        outputs, features = self.net(inputs)
        return outputs, features


class CFairNet_MLP(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, device = 'cuda', in_features=1024, hidden_features=1024):
        super(CFairNet_MLP, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.used_classes = 2 # can only handle binary attributes
        
        mod = import_module("models.basemodels_mlp")
        cusModel = getattr(mod, self.backbone)
        self.net = cusModel(n_classes=self.num_classes, in_features= in_features, hidden_features=hidden_features)
        hidden_size = hidden_features
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [hidden_size] + [adversary_size]
        self.num_adversaries_layers = len([adversary_size])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.used_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.used_classes)])
        
    def forward(self, inputs, labels):
        h_relu = inputs
        outputs, features = self.net(h_relu)
        h_relu = F.relu(features)
        
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        
        for j in range(self.used_classes):
            idx = labels[:, 0] == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        return outputs, c_losses

    def inference(self, inputs):
        outputs, features = self.net(inputs)
        return outputs, features