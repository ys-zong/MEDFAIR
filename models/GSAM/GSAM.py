import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc
from models.basenet import BaseNet
from importlib import import_module
import torchvision

from models.GSAM.utils import GSAM_optimizer, LinearScheduler

    
class GSAM(BaseNet):
    def __init__(self, opt, wandb):
        super(GSAM, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
        

    def set_network(self, opt):
        """Define the network"""
        
        if not self.is_3d:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
            
        else:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        
        #self.network = cusResNet18(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
        

    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.base_optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )
        self.lr_scheduler = LinearScheduler(T_max=opt['T_max'], \
            max_value=optimizer_setting['lr'], min_value=optimizer_setting['lr']*0.01, optimizer=self.base_optimizer)
        self.rho_scheduler = LinearScheduler(T_max=opt['T_max'], max_value=0.04, min_value=0.02)
        self.gsam_optimizer = GSAM_optimizer(params=self.network.parameters(), base_optimizer=self.base_optimizer, model=self.network,\
             gsam_alpha=0.01, rho_scheduler=self.rho_scheduler, adaptive=False)

    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.gsam_optimizer.state_dict(),
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
            
            self.gsam_optimizer.set_closure(self._criterion, images, targets)
            outputs, loss = self.gsam_optimizer.step()
            self.lr_scheduler.step()
            self.gsam_optimizer.update_rho_t()

            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())
    
            train_loss += loss.item()
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': train_loss / (i+1), 'Training AUC': auc / (i+1)})
        
        auc = 100 * auc / no_iter
        train_loss /= no_iter
        
        
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))
        
        self.epoch += 1
        