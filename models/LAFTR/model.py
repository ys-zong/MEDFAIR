import torch
import torch.nn as nn
from importlib import import_module


class LaftrNet(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, pretrained = True, device = 'cuda', model_var = 'laftr-dp'):
        super(LaftrNet, self).__init__()
        
        self.backbone = backbone
        self.model_var = model_var
        self.num_classes = num_classes
        self.used_classes = 2
        
        mod = import_module("models.basemodels")
        cusModel = getattr(mod, self.backbone)
        self.net = cusModel(n_classes=self.num_classes, pretrained=pretrained)
        hidden_size = self.net.body.fc.in_features
        
        self.device = device

        if self.model_var != "laftr-dp":
            self.adv_neurons =  [hidden_size + self.used_classes - 1] \
                + [adversary_size] \
                + [self.used_classes - 1]
        else:
            self.adv_neurons = [hidden_size] + [adversary_size] + [self.used_classes - 1]
        
        self.num_adversaries_layers = len(self.adv_neurons)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for one class label.
        self.discriminator = nn.ModuleList([nn.Linear(self.adv_neurons[i], self.adv_neurons[i + 1])
                                                         for i in range(self.num_adversaries_layers -1)])

    def forward(self, X, Y=None):
        Y_logits, Z = self.net(X)
        if Y is None:
            # for inference
            return Y_logits, Z

        if self.model_var != "laftr-dp":
            Z = torch.cat(
                [Z, torch.unsqueeze(Y[:, 0].type(torch.FloatTensor), 1).to(self.device)],
                axis=1,)
        for hidden in self.discriminator:
            Z = hidden(Z)
        
        # For discriminator loss
        A_logits = torch.squeeze(Z)
        return Y_logits, A_logits

    def inference(self, X):
        Y_logits, Z = self.net(X)
        return Y_logits, Z


class LaftrNet3D(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, pretrained = True, device = 'cuda', model_var = 'laftr-dp'):
        super(LaftrNet3D, self).__init__()
        
        self.backbone = backbone
        self.model_var = model_var
        self.num_classes = num_classes
        self.used_classes = 2
        
        mod = import_module("models.basemodels_3d")
        cusModel = getattr(mod, self.backbone)
        self.net = cusModel(n_classes=self.num_classes, pretrained=pretrained)
        hidden_size = self.net.body.fc.in_features
        
        self.device = device

        if self.model_var != "laftr-dp":
            self.adv_neurons =  [hidden_size + self.used_classes - 1] \
                + [adversary_size] \
                + [self.used_classes - 1]
        else:
            self.adv_neurons = [hidden_size] + [adversary_size] + [self.used_classes - 1]
        
        
        self.num_adversaries_layers = len(self.adv_neurons)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for one class label.
        self.discriminator = nn.ModuleList([nn.Linear(self.adv_neurons[i], self.adv_neurons[i + 1])
                                                         for i in range(self.num_adversaries_layers -1)])

    def forward(self, X, Y=None):
        Y_logits, Z = self.net(X)
        if Y is None:
            # for inference
            return Y_logits, Z
        if self.model_var != "laftr-dp":
            Z = torch.cat(
                [Z, torch.unsqueeze(Y[:, 0].type(torch.FloatTensor), 1).to(self.device)],
                axis=1,)
        for hidden in self.discriminator:
            Z = hidden(Z)
        
        # For discriminator loss
        A_logits = torch.squeeze(Z)
        return Y_logits, A_logits

    def inference(self, X):
        Y_logits, Z = self.net(X)
        return Y_logits, Z



class LaftrNet_MLP(nn.Module):
    def __init__(self, backbone, num_classes, adversary_size = 128, device = 'cuda', model_var = 'laftr-dp', in_features=1024, hidden_features=1024):
        super(LaftrNet_MLP, self).__init__()
        
        self.backbone = backbone
        self.model_var = model_var
        self.num_classes = num_classes
        self.used_classes = 2
        
        mod = import_module("models.basemodels_mlp")
        cusModel = getattr(mod, self.backbone)
        self.net = cusModel(n_classes=self.num_classes, in_features= in_features, hidden_features=hidden_features)
        hidden_size = hidden_features
        
        self.device = device

        if self.model_var != "laftr-dp":
            self.adv_neurons =  [hidden_size + self.used_classes - 1] \
                + [adversary_size] \
                + [self.used_classes - 1]
        else:
            self.adv_neurons = [hidden_size] + [adversary_size] + [self.used_classes - 1]
        
        
        self.num_adversaries_layers = len(self.adv_neurons)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.discriminator = nn.ModuleList([nn.Linear(self.adv_neurons[i], self.adv_neurons[i + 1])
                                                         for i in range(self.num_adversaries_layers -1)])

    def forward(self, X, Y=None):
        Y_logits, Z = self.net(X)
        if Y is None:
            # for inference
            return Y_logits, Z
        if self.model_var != "laftr-dp":
            Z = torch.cat(
                [Z, torch.unsqueeze(Y[:, 0].type(torch.FloatTensor), 1).to(self.device)],
                axis=1,)
        for hidden in self.discriminator:
            Z = hidden(Z)
        
        # For discriminator loss
        A_logits = torch.squeeze(Z)
        return Y_logits, A_logits

    def inference(self, X):
        Y_logits, Z = self.net(X)
        return Y_logits, Z