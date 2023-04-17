import os
from turtle import back
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ODR.model import ODRModel, ODR_Encoder3D, ODRModel3D, ODRModel_MLP
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from models.ODR.utils import OrthoLoss
from utils import basics


class ODR(BaseNet):
    def __init__(self, opt, wandb):
        super(ODR, self).__init__(opt, wandb)
        
        self.set_network(opt)
        self.set_optimizer(opt)

        self.lambda_e = opt['lambda_e']
        self.lambda_od = opt['lambda_od']
        self.gamma_e = opt['gamma_e']
        self.gamma_od = opt['gamma_od']
        self.step_size = opt['step_size']
        
        self.loss = OrthoLoss(self.lambda_e, self.lambda_od, self.gamma_e, self.gamma_od, self.step_size, self.device)
        self.bce = nn.BCEWithLogitsLoss()
        self.cross = nn.CrossEntropyLoss()
        
        
    def set_network(self, opt):
        """Define the network"""
        if self.is_3d:
            self.network = ODRModel3D(backbone = self.backbone, target_classes=self.num_classes, sensitive_classes = self.sens_classes, input_dim = 128, hidden_dim = [256, 128], z_dim = 128, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            self.network = ODRModel_MLP(backbone=self.backbone, target_classes=self.num_classes, sensitive_classes = self.sens_classes, input_dim = 128, hidden_dim = [256, 128], z_dim = 128, in_features = self.in_features, hidden_features = 1024).to(self.device)
        else:
            self.network = ODRModel(backbone = self.backbone, target_classes=self.num_classes, sensitive_classes = self.sens_classes, input_dim = 128, hidden_dim = [256, 128], z_dim = 128, pretrained = self.pretrained).to(self.device)
        

            
    def forward(self, x, targets):
        outputs, c_losses = self.network(x, targets)
        return outputs, c_losses

    def set_optimizer(self, opt):
        optimizer_setting1 = opt['optimizer_setting']
        self.optimizer = optimizer_setting1['optimizer']( 
                            params=self.network.encoder.parameters(), 
                            lr=optimizer_setting1['lr'],
                            weight_decay=optimizer_setting1['weight_decay']
                            )
        
        optimizer_setting2 = opt['optimizer_setting2']
        self.optimizer_2 = optimizer_setting2['optimizer']( 
                            params=self.network.decoder.parameters(), 
                            lr=optimizer_setting2['lr'],
                            weight_decay=optimizer_setting2['weight_decay']
                            )
    
    def _criterion(self, outputs, targets, sensitive, epoch):
        return self.loss(outputs, targets, sensitive, epoch)

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        running_loss = 0.
        auc = 0.
        no_iter = 0
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            
            self.optimizer.zero_grad()
            self.optimizer_2.zero_grad()
            outputs = self.network(images)
            s_zs = outputs[1][2]
            z_t = outputs[2][0]
            
            L_s = self.cross(s_zs, sensitive_attr)
            y_zt = outputs[1][0]
            
            for param in self.network.encoder.backbone.parameters():
                param.requires_grad=False
            L_s.backward(retain_graph=True)
            
            for param in self.network.encoder.backbone.parameters():
                param.requires_grad=True
                
            loss = self._criterion(outputs, targets, sensitive_attr, self.epoch)

            loss.backward()
            
            self.optimizer.step()
            self.optimizer_2.step()
            
            running_loss += loss.item()
            
            auc += calculate_auc(F.sigmoid(y_zt).cpu().data.numpy(), targets.cpu().data.numpy())
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})

        running_loss /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: cls loss:{}'.format(self.epoch, running_loss))
        self.epoch += 1
    
    
    def _val(self, loader):

        self.network.eval()
        
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        val_loss, auc = 0., 0.
        no_iter = 0
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
                outputs = self.network(images)
                
                z_t = outputs[2][0]
                s_zs = outputs[1][2]
                y_zt = outputs[1][0]
                
                loss = self._criterion(outputs, targets, sensitive_attr, self.epoch)
                val_loss += loss.item()
                
                tol_output += F.sigmoid(y_zt).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
                auc += calculate_auc(F.sigmoid(y_zt).cpu().data.numpy(),
                                               targets.cpu().data.numpy())
                no_iter += 1
                
                if self.log_freq and (i % self.log_freq == 0):
                    self.wandb.log({'Validation loss': val_loss / (i+1), 'Validation AUC': auc / (i+1)})

        auc = 100 * auc / no_iter
        val_loss /= no_iter
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        return val_loss, auc, log_dict, pred_df
    
    def _test(self, loader):
        """Compute model output on testing set"""

        self.network.eval()
        
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        tol_features = []
    
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs = self.network(images)
                
                z_t = outputs[2][0]
                s_zs = outputs[1][2]
                y_zt = outputs[1][0]
    
                tol_output += F.sigmoid(y_zt).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, 'pred.csv'), index = False)
        #basics.save_results(t_predictions, tol_target, s_prediction, tol_sensitive, self.save_path, self.experiment)
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
            
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')   

        return log_dict
