from models.basemodels_mlp import cusMLP
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.models.feature_extraction import create_feature_extractor
from importlib import import_module
import numpy as np


# modify the resnet
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class cusResNet18_ODR(nn.Module):    
    def __init__(self, input_dim, pretrained = True):
        super(cusResNet18_ODR, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.fc = Identity()
        
        self.avgpool = resnet.avgpool
        
        self.returnkey_avg = 'avgpool'
        #self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg})
        
        hidden_size = 512
        self.encoder1_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

        self.encoder2_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

    def forward(self, x):
        outputs = self.body(x)[self.returnkey_avg]
        x1 = self.encoder1_net(outputs)
        x2 = self.encoder2_net(outputs)
        return x1, x2

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()


class Tabular_ModelDecoder(nn.Module):
    
    def __init__(self, z_dim, hidden_dims, target_classes, sensitive_classes):

        super(Tabular_ModelDecoder, self).__init__()

        #List of layers excluding the output layer
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes

        #import pdb; pdb.set_trace()
        self.num_layers = [self.z_dim] + self.hidden_dims
        
        #Activation function
        self.act_f = nn.ReLU()

        #Architecture of the decoder 1
        self.layers_1 = []
        for layer_index_1 in range(1, len(self.num_layers)):
            self.layers_1 += [nn.Linear(self.num_layers[layer_index_1 - 1],
                            self.num_layers[layer_index_1]), self.act_f]
        
        #Architecture of the decoder 2
        self.layers_2 = []
        for layer_index_2 in range(1, len(self.num_layers)):
            self.layers_2 += [nn.Linear(self.num_layers[layer_index_2 - 1],
                            self.num_layers[layer_index_2]), self.act_f]
        
        #Output layer
        self.output_1 = nn.Linear(self.num_layers[-1], self.target_classes)
        self.output_2 = nn.Linear(self.num_layers[-1], self.sensitive_classes)
        
        self.Decoder_1 = nn.ModuleList(self.layers_1)
        self.Decoder_2 = nn.ModuleList(self.layers_2)
    
    def forward(self, z_1, z_2):

        out_1 = z_1
        for layers_1 in self.Decoder_1:
            out_1 = layers_1(out_1)
        y_zt = self.output_1(out_1)
            
        out_1 = z_1
        out_2 = z_2
        for layers_2 in self.Decoder_2:
            out_1 = layers_2(out_1)
            out_2 = layers_2(out_2)
        s_zt = self.output_2(out_1)
        s_zs = self.output_2(out_2)
        
        return y_zt, s_zt, s_zs
    
    
class ODR_Encoder(nn.Module):

    def __init__(self, input_dim, z_dim, pretrained):

        super(ODR_Encoder, self).__init__()
        
        self.z_dim = z_dim
        
        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(input_dim, z_dim)
        self.log_var_1      = nn.Linear(input_dim, z_dim)

        self.mean_encoder_2 = nn.Linear(input_dim, z_dim)
        self.log_var_2      = nn.Linear(input_dim, z_dim)

        #Activation function
        self.act_f = nn.ReLU()

        self.backbone = cusResNet18_ODR(input_dim, pretrained = pretrained)

    def forward(self, x):
        
        out_1, out_2 = self.backbone(x)
        
        mean_t = self.mean_encoder_1(self.act_f(out_1))
        log_var_t = self.log_var_1(self.act_f(out_1))
        
        mean_s = self.mean_encoder_2(self.act_f(out_2))
        log_var_s = self.log_var_2(self.act_f(out_2))
        
        return mean_t, mean_s, log_var_t, log_var_s

    
class ODRModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, target_classes, sensitive_classes, backbone, pretrained = True):
        super().__init__()

        self.encoder = ODR_Encoder(input_dim, z_dim, pretrained)
        self.decoder = Tabular_ModelDecoder(z_dim, hidden_dim, target_classes, sensitive_classes)
    
    def mean_tensors(self, mean_1, mean_2, i):
        mean_1[i] = 1
        mean_2[i] = 0
        mean_t = torch.from_numpy(mean_1).float()
        mean_s = torch.from_numpy(mean_2).float()
        return mean_t, mean_s
    
    def reparameterization_ODR(self, mean_t, mean_s, log_var_t, log_var_s):

        mean_1, mean_2 = self.mean_tensors(np.zeros(128), np.ones(128), 13)
        if mean_t.is_cuda:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)).cuda())
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)).cuda())
        else:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)))
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)))
        return z1, z2
    
    def forward(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = self.reparameterization_ODR(mean_t, mean_s, log_var_t, log_var_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        return (mean_t, mean_s, log_var_t, log_var_s), (y_zt, s_zt, s_zs), (z1, z2)

    def inference(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = self.reparameterization_ODR(mean_t, mean_s, log_var_t, log_var_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        
        return y_zt, z1


######## 3D 


class cusResNet18_3d_ODR(nn.Module):    
    def __init__(self, input_dim, pretrained = True):
        super(cusResNet18_3d_ODR, self).__init__()
        resnet = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        resnet.fc = Identity()
        
        self.avgpool = resnet.avgpool
        
        self.returnkey_avg = 'avgpool'
        #self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg})
        
        hidden_size = 512
        self.encoder1_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

        self.encoder2_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

    def forward(self, x):
        outputs = self.body(x)[self.returnkey_avg]
        x1 = self.encoder1_net(outputs)
        x2 = self.encoder2_net(outputs)
        return x1, x2

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()


class Tabular_ModelDecoder3D(nn.Module):
    
    def __init__(self, z_dim, hidden_dims, target_classes, sensitive_classes):

        super(Tabular_ModelDecoder3D, self).__init__()

        #List of layers excluding the output layer
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes

        #import pdb; pdb.set_trace()
        self.num_layers = [self.z_dim] + self.hidden_dims
        
        #Activation function
        self.act_f = nn.ReLU()

        #Architecture of the decoder 1
        self.layers_1 = []
        for layer_index_1 in range(1, len(self.num_layers)):
            self.layers_1 += [nn.Linear(self.num_layers[layer_index_1 - 1],
                            self.num_layers[layer_index_1]), self.act_f]
        
        #Architecture of the decoder 2
        self.layers_2 = []
        for layer_index_2 in range(1, len(self.num_layers)):
            self.layers_2 += [nn.Linear(self.num_layers[layer_index_2 - 1],
                            self.num_layers[layer_index_2]), self.act_f]
        
        #Output layer
        self.output_1 = nn.Linear(self.num_layers[-1], self.target_classes)
        self.output_2 = nn.Linear(self.num_layers[-1], self.sensitive_classes)
        
        self.Decoder_1 = nn.ModuleList(self.layers_1)
        self.Decoder_2 = nn.ModuleList(self.layers_2)
    
    def forward(self, z_1, z_2):

        out_1 = z_1
        for layers_1 in self.Decoder_1:
            out_1 = layers_1(out_1)
        y_zt = self.output_1(out_1)
            
        out_1 = z_1
        out_2 = z_2
        for layers_2 in self.Decoder_2:
            out_1 = layers_2(out_1)
            out_2 = layers_2(out_2)
        s_zt = self.output_2(out_1)
        s_zs = self.output_2(out_2)
        
        return y_zt, s_zt, s_zs
    
    
class ODR_Encoder3D(nn.Module):

    def __init__(self, input_dim, z_dim, pretrained):

        super(ODR_Encoder3D, self).__init__()
        
        self.z_dim = z_dim
        
        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(input_dim, z_dim)
        self.log_var_1      = nn.Linear(input_dim, z_dim)

        self.mean_encoder_2 = nn.Linear(input_dim, z_dim)
        self.log_var_2      = nn.Linear(input_dim, z_dim)

        #Activation function
        self.act_f = nn.ReLU()

        self.backbone = cusResNet18_3d_ODR(input_dim, pretrained = pretrained)

    def forward(self, x):
        
        out_1, out_2 = self.backbone(x)
        
        mean_t = self.mean_encoder_1(self.act_f(out_1))
        log_var_t = self.log_var_1(self.act_f(out_1))
        
        mean_s = self.mean_encoder_2(self.act_f(out_2))
        log_var_s = self.log_var_2(self.act_f(out_2))
        
        return mean_t, mean_s, log_var_t, log_var_s

    
class ODRModel3D(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, target_classes, sensitive_classes, backbone, pretrained = True):
        super().__init__()

        self.encoder = ODR_Encoder3D(input_dim, z_dim, pretrained)
        self.decoder = Tabular_ModelDecoder3D(z_dim, hidden_dim, target_classes, sensitive_classes)
    
    def mean_tensors(self, mean_1, mean_2, i):
        mean_1[i] = 1
        mean_2[i] = 0
        mean_t = torch.from_numpy(mean_1).float()
        mean_s = torch.from_numpy(mean_2).float()
        return mean_t, mean_s
    
    def reparameterization_ODR(self, mean_t, mean_s, log_var_t, log_var_s):

        mean_1, mean_2 = self.mean_tensors(np.zeros(128), np.ones(128), 13)
        if mean_t.is_cuda:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)).cuda())
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)).cuda())
        else:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)))
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)))
        return z1, z2
    
    def forward(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = self.reparameterization_ODR(mean_t, mean_s, log_var_t, log_var_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        return (mean_t, mean_s, log_var_t, log_var_s), (y_zt, s_zt, s_zs), (z1, z2)




################ MLP
class cusMLP_ODR(nn.Module):    
    def __init__(self, input_dim, in_features, hidden_features):
        super(cusMLP_ODR, self).__init__()
        self.backbone = cusMLP(n_classes=1, in_features= in_features, hidden_features=hidden_features)
        self.backbone.backbone.fc2 = Identity()
        
        hidden_size = self.backbone.backbone.fc1.out_features
        self.encoder1_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

        self.encoder2_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, input_dim))

    def forward(self, x):
        y, hidden = self.backbone(x)
        x1 = self.encoder1_net(hidden)
        x2 = self.encoder2_net(hidden)
        return x1, x2

    def inference(self, x):
        y, hidden = self.backbone(x)
        return y, hidden


class Tabular_ModelDecoder_MLP(nn.Module):
    
    def __init__(self, z_dim, hidden_dims, target_classes, sensitive_classes):

        super(Tabular_ModelDecoder_MLP, self).__init__()

        #List of layers excluding the output layer
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes

        #import pdb; pdb.set_trace()
        self.num_layers = [self.z_dim] + self.hidden_dims
        
        #Activation function
        self.act_f = nn.ReLU()

        #Architecture of the decoder 1
        self.layers_1 = []
        for layer_index_1 in range(1, len(self.num_layers)):
            self.layers_1 += [nn.Linear(self.num_layers[layer_index_1 - 1],
                            self.num_layers[layer_index_1]), self.act_f]
        
        #Architecture of the decoder 2
        self.layers_2 = []
        for layer_index_2 in range(1, len(self.num_layers)):
            self.layers_2 += [nn.Linear(self.num_layers[layer_index_2 - 1],
                            self.num_layers[layer_index_2]), self.act_f]
        
        #Output layer
        self.output_1 = nn.Linear(self.num_layers[-1], self.target_classes)
        self.output_2 = nn.Linear(self.num_layers[-1], self.sensitive_classes)
        
        self.Decoder_1 = nn.ModuleList(self.layers_1)
        self.Decoder_2 = nn.ModuleList(self.layers_2)
    
    def forward(self, z_1, z_2):

        out_1 = z_1
        for layers_1 in self.Decoder_1:
            out_1 = layers_1(out_1)
        y_zt = self.output_1(out_1)
            
        out_1 = z_1
        out_2 = z_2
        for layers_2 in self.Decoder_2:
            out_1 = layers_2(out_1)
            out_2 = layers_2(out_2)
        s_zt = self.output_2(out_1)
        s_zs = self.output_2(out_2)
        
        return y_zt, s_zt, s_zs
    
    
class ODR_Encoder_MLP(nn.Module):

    def __init__(self, input_dim, z_dim, in_features, hidden_features):

        super(ODR_Encoder_MLP, self).__init__()
        
        self.z_dim = z_dim
        
        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(input_dim, z_dim)
        self.log_var_1      = nn.Linear(input_dim, z_dim)

        self.mean_encoder_2 = nn.Linear(input_dim, z_dim)
        self.log_var_2      = nn.Linear(input_dim, z_dim)

        #Activation function
        self.act_f = nn.ReLU()

        self.backbone = cusMLP_ODR(input_dim, in_features, hidden_features)

    def forward(self, x):
        
        out_1, out_2 = self.backbone(x)
        
        mean_t = self.mean_encoder_1(self.act_f(out_1))
        log_var_t = self.log_var_1(self.act_f(out_1))
        
        mean_s = self.mean_encoder_2(self.act_f(out_2))
        log_var_s = self.log_var_2(self.act_f(out_2))
        
        return mean_t, mean_s, log_var_t, log_var_s

    
class ODRModel_MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim, target_classes, sensitive_classes, backbone, in_features, hidden_features):
        super().__init__()

        self.encoder = ODR_Encoder_MLP(input_dim, z_dim, in_features, hidden_features)
        self.decoder = Tabular_ModelDecoder_MLP(z_dim, hidden_dim, target_classes, sensitive_classes)
    
    def mean_tensors(self, mean_1, mean_2, i):
        mean_1[i] = 1
        mean_2[i] = 0
        mean_t = torch.from_numpy(mean_1).float()
        mean_s = torch.from_numpy(mean_2).float()
        return mean_t, mean_s
    
    def reparameterization_ODR(self, mean_t, mean_s, log_var_t, log_var_s):

        mean_1, mean_2 = self.mean_tensors(np.zeros(128), np.ones(128), 13)
        if mean_t.is_cuda:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)).cuda())
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)).cuda())
        else:
            z1 = mean_t + (torch.exp(log_var_t/2) @ torch.normal(mean_1, torch.eye(128)))
            z2 = mean_s + (torch.exp(log_var_s/2) @ torch.normal(mean_2, torch.eye(128)))
        return z1, z2
    
    def forward(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = self.reparameterization_ODR(mean_t, mean_s, log_var_t, log_var_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        return (mean_t, mean_s, log_var_t, log_var_s), (y_zt, s_zt, s_zs), (z1, z2)

    def inference(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = self.reparameterization_ODR(mean_t, mean_s, log_var_t, log_var_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        
        return y_zt, z1