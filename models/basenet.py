import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import os
import random
from datetime import datetime
from models.utils import standard_val, standard_test
from datasets.utils import get_dataset
from utils.evaluation import calculate_metrics, calculate_FPR_FNR
from utils import basics


class BaseNet(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, opt, wandb):
        super(BaseNet, self).__init__()
        self.opt = opt
        self.epoch = 0
        self.experiment = opt['experiment']
        self.device = opt['device']
        self.test_mode = opt['test_mode']
        
        self.save_path = opt['save_folder']
        self.resume_path = opt['resume_path']
        self.log_freq = opt['log_freq']
        self.pretrained = opt['pretrained']
        
        self.num_classes = opt['num_classes']
        self.used_classes = opt['sens_classes']
        self.sens_classes = opt['sens_classes']
        self.output_dim = opt['output_dim']
        self.wandb = wandb
        self.hyper_search = opt['hyper_search']
        self.hash = opt['hash']
        
        self.seed = opt['random_seed']
        self.set_random_seed(self.seed)
        
        self.val_strategy = opt['val_strategy']
        
        self.best_val_acc = 0.
        self.best_val_loss = float("inf")
        self.best_worst_auc = 0.
        self.best_log_dict = {}
        self.early_stopping = opt['early_stopping']
        self.patience = 0
        
        self.backbone = opt['backbone']
        
        self.is_tabular = opt['is_tabular']
        self.is_3d = opt['is_3d']
        if self.is_3d:
            self.input_size = opt['input_size']
            self.sample_duration = opt['sample_duration']
        
        self.dataset_name = opt['dataset_name']
        self.bianry_train_multi_test = opt['bianry_train_multi_test']
        
        self.set_data(opt)
        if self.is_tabular:
            #self.in_features = len(self.train_data.data_df.columns) - 1
            #self.in_features = 10000
            self.in_features = 146
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.cross_testing = opt['cross_testing']
        if self.cross_testing:
            self.load_path = opt['cross_testing_model_path_single']
    
    def set_data(self, opt):
        """Set up the dataloaders"""
        self.train_data, self.val_data, self.test_data, self.train_loader, self.val_loader, self.test_loader, self.val_meta, self.test_meta = get_dataset(opt)

    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
            params=filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )

    def _criterion(self, output, target):
        return self.criterion(output, target)

    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def freeze(self):
        """Stops gradient computation"""
        for para in self.network.parameters():
            para.requires_grad = False

    def activate(self):
        """Activates gradient computation"""
        for para in self.network.parameters():
            para.requires_grad = True

    def set_random_seed(self, seed_number):
        # position of setting seeds also matters
        os.environ['PYTHONHASHSEED'] = str(seed_number)
        np.random.seed(seed_number)
        random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.random.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def log_wandb(self, log_dict):
        self.wandb.log(log_dict)
    
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        auc, val_loss, log_dict, pred_df = standard_val(self.opt, self.network, loader, self._criterion, self.sens_classes, self.wandb)
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
    
    def _test(self, loader):
        """Compute model output on testing set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = standard_test(self.opt, self.network, loader, self._criterion, self.wandb)

        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, 'pred.csv'), index = False)
        
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
        
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        
        return log_dict
    
    def train(self, epoch):
        # Train the model for one epoch, evaluate on validation set and save the best model

        start_time = datetime.now()
        self._train(self.train_loader)
        # basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        val_loss, val_auc, log_dict, pred_df = self._val(self.val_loader)
        self.patience += 1
        
        val_flag = False
        if self.val_strategy == 'loss':
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                val_flag = True
                
        elif self.val_strategy == 'worst_auc':
            worst_auc = log_dict['worst_auc']
            worst_group = log_dict['worst_group']
            if worst_auc > self.best_worst_auc:
                self.best_worst_auc = worst_auc
                val_flag = True
                print('The worst group is {} with AUC: {}'.format(worst_group, worst_auc))
        if val_flag:
            self.best_log_dict = log_dict
            self.best_pred_df = pred_df
            if self.hyper_search is True:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
                print('saving best model in epoch ', epoch, ' in ', os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
            else:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, str(self.seed) + '_best.pth'))
                print('saving best model in epoch ', epoch, ' in ', os.path.join(self.save_path, str(self.seed) + '_best.pth'))
            self.patience = 0

        duration = datetime.now() - start_time
        print('Finish training epoch {}, Val AUC: {}, time used: {}'.format(self.epoch, val_auc, duration))
        if self.patience >= self.early_stopping:
            return True
        else:
            return False
    
    def test(self):
        
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

        log_dict = self._test(self.test_loader)
        
        print('Finish testing, testing performance: ')
        print(log_dict)
        
        return pd.DataFrame(log_dict, index=[0])
    
    def record_val(self):
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(self.best_pred_df, self.val_meta, self.opt)
        self.best_log_dict['Overall FPR'] = overall_FPR
        self.best_log_dict['Overall FNR'] = overall_FNR
        for i, FPR in enumerate(FPRs):
            self.best_log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            self.best_log_dict['FNR-group_' + str(i)] = FNR
        log_dict = basics.add_dict_prefix(self.best_log_dict, 'Val ')
        print('Validation performance: ', log_dict)
        
        return pd.DataFrame(log_dict, index=[0])