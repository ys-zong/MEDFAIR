import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc
from models.basenet import BaseNet
from importlib import import_module

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.SAM.utils import SAM_optimizer, disable_running_stats, enable_running_stats

    
class SAM(BaseNet):
    def __init__(self, opt, wandb):
        super(SAM, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
    
    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features= self.in_features, hidden_features = 1024).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
       
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.base_optimizer = torch.optim.Adam
        self.optimizer = SAM_optimizer(params = self.network.parameters(), base_optimizer = self.base_optimizer, rho=opt['rho'], adaptive=opt['adaptive'], lr=optimizer_setting['lr'], weight_decay=optimizer_setting['weight_decay'])
        
        self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, T_max=opt['T_max'])

    def _criterion(self, output, target):
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        return self.criterion(output, target)


    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict
        
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        
        train_loss = 0
        auc = 0.
        no_iter = 0
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            
            enable_running_stats(self.network)
            outputs, _ = self.network(images)
    
            loss = self._criterion(outputs, targets)
            loss.mean().backward()
            self.optimizer.first_step(zero_grad=True)
            self.scheduler.step()
            
            disable_running_stats(self.network)
            outputs, _ = self.network(images)
            self._criterion(outputs, targets).mean().backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())
    
            train_loss += loss.mean().item()
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': train_loss / (i+1), 'Training AUC': auc / (i+1)})
        
        auc = 100 * auc / no_iter
        train_loss /= no_iter
        
        
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))
        
        self.epoch += 1
        