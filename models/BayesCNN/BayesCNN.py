import torch
import torch.nn.functional as F
from utils import basics
from utils.predictions import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from importlib import import_module
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss


class BayesCNN(BaseNet):
    def __init__(self, opt, wandb):
        super(BayesCNN, self).__init__(opt, wandb)
        
        
        self.const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
        }
        
        self.set_network(opt)
        self.set_optimizer(opt)
        self.batch_size = opt['batch_size']
        
        self.num_monte_carlo = opt['num_monte_carlo']

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
        
        dnn_to_bnn(self.network, self.const_bnn_prior_parameters)
        self.network.to(self.device)

    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
            params=filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )
    
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        
        train_loss, auc, no_iter = 0, 0, 0
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _ = self.network(images)
    
            loss = self._criterion(outputs, targets)
            kl = get_kl_loss(self.network)
            loss = loss + kl / self.batch_size 
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
    
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        val_loss, auc, worst_auc = 0., 0., 0.
        no_iter = 0
        with torch.no_grad():
            for i, (index, images, targets, sensitive_attr) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                
                output_mc = []
                for mc_run in range(self.num_monte_carlo):
                    outputs, _ = self.network(images)
                    output_mc.append(outputs)
                outputs = torch.stack(output_mc).mean(dim=0) 
                
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
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
    
    def _test(self, loader):
        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
        with torch.no_grad():
            for i, (index, images, targets, sensitive_attr) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                
                output_mc = []
                for mc_run in range(self.num_monte_carlo):
                    outputs, _ = self.network(images)
                    output_mc.append(outputs)
                outputs = torch.stack(output_mc).mean(dim=0)
                
                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                
        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
        
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        
        return log_dict