import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.SWA import SWA
from importlib import import_module

#from torch.optim.swa_utils import AveragedModel, SWALR
from models.SWAD.utils import AveragedModel, update_bn, LossValley
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.SAM.utils import SAM_optimizer
from torch.nn.modules.batchnorm import _BatchNorm
    

class SAMSWAD(SWA):
    def __init__(self, opt, wandb):
        super(SAMSWAD, self).__init__(opt, wandb)
        self.set_network(opt)

        self.annealing_epochs = opt['swa_annealing_epochs']
        
        self.set_optimizer(opt)
        self.swad = LossValley(n_converge = opt['swad_n_converge'], n_tolerance = opt['swad_n_converge'] + opt['swad_n_tolerance'], 
                               tolerance_ratio = opt['swad_tolerance_ratio'])
        
        self.step = 0
        

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
        
        self.swad_model = AveragedModel(self.network).to(self.device)

    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def state_dict(self):
        state_dict = {
            'model': self.swad_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict
    
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.base_optimizer = torch.optim.Adam

        self.optimizer = SAM_optimizer(params = self.network.parameters(), base_optimizer = self.base_optimizer, rho=opt['rho'], adaptive=opt['adaptive'], lr=optimizer_setting['lr'], weight_decay=optimizer_setting['weight_decay'])
        
        self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, T_max=opt['T_max'])
        
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        #self.swa_scheduler = SWALR(self.optimizer, anneal_epochs = self.annealing_epochs, swa_lr=self.swa_lr)

    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        
        train_loss = 0
        auc = 0.
        no_iter = 0
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            
            outputs, _ = self.network(images)
    
            loss = self._criterion(outputs, targets)
            loss.mean().backward()
            self.optimizer.first_step(zero_grad=True)
            self.scheduler.step()
            
            #disable_running_stats(self.network)
            outputs, _ = self.network(images)
            self._criterion(outputs, targets).mean().backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()
            
            self.step += 1
            self.swad_model.update_parameters(self.network, step = self.step)
    
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
        
            
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        val_loss, auc, worst_auc = 0., 0., 0.
        no_iter = 0
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                loss = self._criterion(outputs, targets)
                val_loss += loss.item()
                
                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
                auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                               targets.cpu().data.numpy())
                
                no_iter += 1
                
                
                if self.log_freq and (i % self.log_freq == 0):
                    self.wandb.log({'Validation loss': val_loss / (i+1), 'Validation AUC': auc / (i+1)})
    
        auc = 100 * auc / no_iter
        val_loss /= no_iter
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        
        self.swad.update_and_evaluate(self.swad_model, 1-log_dict['worst_auc']) #use 1 minus to reverse the trend
        
        if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
            print("SWAD valley is dead -> early stop !")
            self.patience = -1
        self.swad_model = AveragedModel(self.network)
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
    
    
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
            self.swad_model.load_state_dict(state_dict['model'])
            #update_bn(self.train_loader, self.swad_model, device = self.device, is_testing=True) 
            #self.network = self.swad_model.to(self.device)
        else:
            self.swad_model = self.swad.get_final_model()
            update_bn(self.train_loader, self.swad_model, device = self.device) 
            if self.hyper_search is True:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
                print('saving best model in ', os.path.join(self.save_path, self.hash + '_' + str(self.seed) + '_best.pth'))
            else:
                basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, str(self.seed) + '_best.pth'))
                print('saving best model in ', os.path.join(self.save_path, str(self.seed) + '_best.pth'))
            self.network = self.swad_model.to(self.device)

        log_dict = self._test(self.test_loader)

        print('Finish testing')
        print(log_dict)
        return pd.DataFrame(log_dict, index=[0])
    
    
    def train(self, epoch):
        # Train the model for one epoch, evaluate on validation set and save the best model

        start_time = datetime.now()
        self._train(self.train_loader)
        # basics.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        val_loss, val_auc, log_dict, pred_df = self._val(self.val_loader)
        if self.patience != -1:
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
        if val_flag and (self.patience != -1):
            self.best_log_dict = log_dict
            self.best_pred_df = pred_df
            
            self.patience = 0

        duration = datetime.now() - start_time
        print('Finish training epoch {}, Val AUC: {}, time used: {}'.format(self.epoch, val_auc, duration))
        if self.patience >= self.early_stopping or self.patience == -1:
            return True
        else:
            return False
        
    def _test(self, loader):
        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, _ = self.swad_model(images)
                
                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
        
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        #pred_df.to_csv(os.path.join(self.save_path, 'pred.csv'), index = False)
        #basics.save_results(t_predictions, tol_target, s_prediction, tol_sensitive, self.save_path)
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
        
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        #log_dict.update({'s_acc': round(sens_acc, 4),})
        
        return log_dict