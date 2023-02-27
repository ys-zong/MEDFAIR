import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CFair.model import CFairNet, CFairNet3D, CFairNet_MLP
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from models.utils import standard_val, standard_test



class CFair(BaseNet):
    def __init__(self, opt, wandb):
        super(CFair, self).__init__(opt, wandb)

        self.test_classes = opt['sens_classes']
        self.sens_classes = 2

        self.set_network(opt)
        self.set_optimizer(opt)

        self.mu = opt['mu']  # coefficient for adversarial loss

    def set_network(self, opt):
        """Define the network"""
        if self.is_3d:
            self.network = CFairNet3D(backbone = self.backbone, num_classes=self.num_classes, adversary_size = 128, pretrained = self.pretrained).to(self.device)  
        elif self.is_tabular:
            self.network = CFairNet_MLP(backbone = self.backbone, num_classes=self.num_classes, adversary_size=128, device=self.device, in_features=self.in_features, hidden_features=1024).to(self.device)  
        else:
            self.network = CFairNet(backbone = self.backbone, num_classes=self.num_classes, adversary_size = 128, pretrained = self.pretrained).to(self.device)

    def get_reweight_tensor(self, model_name):
        train_target_attrs = self.train_data.A
        train_target_labels = self.train_data.Y
        train_y_1 = np.mean(train_target_labels)
        
        if model_name == "cfair":
            reweight_target_tensor = torch.FloatTensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).to(self.device)
        elif model_name == "cfair-eo":
            reweight_target_tensor = torch.FloatTensor([1.0, 1.0]).to(self.device)
        
        train_idx = train_target_attrs == 0
        train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(train_target_labels[~train_idx])
        reweight_attr_0_tensor = torch.FloatTensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).to(self.device)
        reweight_attr_1_tensor = torch.FloatTensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).to(self.device)
        reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]
        return reweight_target_tensor, reweight_attr_tensors

    def _train(self, loader):
        """Train the model for one epoch"""
        reweight_target_tensor, reweight_attr_tensors = self.get_reweight_tensor(model_name='cfair')
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=reweight_target_tensor)
        self.network.train()
        
        running_loss = 0.
        running_adv_loss = 0.
        auc = 0.
        no_iter = 0
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            self.optimizer.zero_grad()
            ypreds, apreds = self.network.forward(images, targets)
            
            loss = self._criterion(ypreds, targets)
            
            adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], sensitive_attr[targets[:, 0] == j], weight= reweight_attr_tensors[j]) for j in range(self.sens_classes)]))
            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            
            loss += self.mu * adv_loss
            
            loss.backward()
            self.optimizer.step()
            
            auc += calculate_auc(F.sigmoid(ypreds).cpu().data.numpy(), targets.cpu().data.numpy())
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})

        running_loss /= no_iter
        running_adv_loss /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: cls loss:{}, adv loss:{}'.format(self.epoch, running_loss, running_adv_loss))
        self.epoch += 1
    
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        auc, val_loss, log_dict, pred_df = standard_val(self.opt, self.network, loader, self._criterion, self.test_classes, self.wandb)
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
    
    def _test(self, loader):
        """Compute model output on testing set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = standard_test(self.opt, self.network, loader, self._criterion, self.wandb)
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.test_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
        
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        self.opt['sens_classes'] = 2
        return log_dict