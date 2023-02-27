import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import basemodels
from utils import basics
import pandas as pd
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
import torchvision

from importlib import import_module
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

    
class SWA(BaseNet):
    def __init__(self, opt, wandb):
        super(SWA, self).__init__(opt, wandb)
        self.set_network(opt)
        self.swa_start = opt['swa_start']
        self.swa_lr = opt['swa_lr']
        self.annealing_epochs = opt['swa_annealing_epochs']
        
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
        
        self.swa_model = AveragedModel(self.network).to(self.device)

    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
            params=filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        self.swa_scheduler = SWALR(self.optimizer, anneal_epochs = self.annealing_epochs, swa_lr=self.swa_lr)

    def state_dict(self):
        state_dict = {
            'model': self.swa_model.state_dict(),
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
            
            self.optimizer.zero_grad()
            outputs, _ = self.network(images)
    
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
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
        
        if self.epoch >= self.swa_start:
            #if self.epoch == self.swa_start:
            #    for g in self.optimizer.param_groups:
            #        g['lr'] = 0.05
                    
            self.swa_model.update_parameters(self.network)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()


            
    def test(self):
        if self.test_mode:
            if not self.cross_testing:
                if self.hyper_search is True:
                    state_dict = torch.load(os.path.join(self.resume_path, self.hash + '_' + str(self.seed) + '_best.pth'))
                    print('Testing, loaded model from ', os.path.join(self.resume_path,  self.hash + '_' + str(self.seed) + '_best.pth'))
                else:
                    state_dict = torch.load(os.path.join(self.resume_path, str(self.seed) +'_best.pth'))
                    print('Testing, loaded model from ', os.path.join(self.resume_path, str(self.seed) +'_best.pth'))
            else:
                state_dict = torch.load(self.load_path)
                print('Testing, loaded model from ', self.load_path)
            self.network.load_state_dict(state_dict['model'])
        else:
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device = self.device) 
            if self.hyper_search is True:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
                print('saving best model in ', os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
            else:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, str(self.seed) + '_best.pth'))
                print('saving best model in ', os.path.join(self.save_path, str(self.seed) + '_best.pth'))
            self.network = self.swa_model.to(self.device)

        log_dict = self._test(self.test_loader)

        print('Finish testing')
        print(log_dict)
        return pd.DataFrame(log_dict, index=[0])
        
        
    def _test(self, loader):
        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, _ = self.swa_model(images)
                
                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, 'pred.csv'), index = False)
        #basics.save_results(t_predictions, tol_target, s_prediction, tol_sensitive, self.save_path)
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
        
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        #log_dict.update({'s_acc': round(sens_acc, 4),})
        
        return log_dict