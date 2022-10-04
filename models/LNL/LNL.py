import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from models.LNL.model import LNLNet, LNLNet3D, LNLNet_MLP, LNLPredictor_MLP, LNLPredictor, LNLPredictor3D, grad_reverseLNL
from utils.evaluation import calculate_auc
from models.basenet import BaseNet


class LNL(BaseNet):
    def __init__(self, opt, wandb):
        super(LNL, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
        self._lambda = opt['_lambda']
        
        self.pred_loss = nn.CrossEntropyLoss()

    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            self.network = LNLNet3D(backbone = self.backbone, num_classes=self.num_classes, pretrained=self.pretrained).to(self.device)
            #pred_ch = self.network.body.layer2[-1].conv1[0].in_channels
            pred_ch = pred_ch = self.network.pred_ch
            self.pred_net = LNLPredictor3D(input_ch=pred_ch, num_classes=self.sens_classes).to(self.device)
        elif self.is_tabular:
            self.network = LNLNet_MLP(backbone = self.backbone, num_classes=self.num_classes, in_features=self.in_features, hidden_features=1024).to(self.device)
            pred_ch = self.network.pred_ch
            self.pred_net = LNLPredictor_MLP(input_ch=pred_ch, num_classes=self.sens_classes).to(self.device)

        else:
            self.network = LNLNet(backbone = self.backbone, num_classes=self.num_classes, pretrained=self.pretrained).to(self.device)
            #pred_ch = self.network.body.layer2[-1].conv1.in_channels
            pred_ch = self.network.pred_ch
            self.pred_net = LNLPredictor(input_ch=pred_ch, num_classes=self.sens_classes).to(self.device)
        
        
        #print(self.network)
        #print(self.pred_net)
        
    def forward(self, x):
        pred_label, feat_label  = self.network(x)
        return pred_label, feat_label
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
                            params=filter(lambda p: p.requires_grad, self.network.parameters()), 
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        self.optimizer_pred = optimizer_setting['optimizer'](
            params=filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay'])

        lr_lambda = lambda step: opt['lr_decay_rate'] ** (step // opt['lr_decay_period'])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_pred = optim.lr_scheduler.LambdaLR(self.optimizer_pred, lr_lambda=lr_lambda, last_epoch=-1)

    def _train(self, loader):
        """Train the model for one epoch"""
        self.network.train()

        running_loss = 0.
        running_adv_loss = 0.
        running_MI = 0.
        auc = 0.
        no_iter = 0
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            self.optimizer.zero_grad()

            self.optimizer.zero_grad()
            self.optimizer_pred.zero_grad()

            pred_label, feat_label = self.forward(images)
            pseudo_pred, _ = self.pred_net(feat_label)
            
            loss_pred_cls = self._criterion(pred_label, targets)
            pseudo_pred = F.sigmoid(pseudo_pred)
            loss_pseudo_pred = torch.mean(torch.sum(pseudo_pred * torch.log(pseudo_pred), 1))

            loss = loss_pred_cls + loss_pseudo_pred * self._lambda
            loss.backward()

            self.optimizer.step()
            self.optimizer_pred.step()

            self.optimizer.zero_grad()
            self.optimizer_pred.zero_grad()

            pred_label, feat_label = self.forward(images)
            feat_sens = grad_reverseLNL(feat_label)
            _, pred_ = self.pred_net(feat_sens)
            loss_pred_sensi = self.pred_loss(pred_, sensitive_attr)
            loss_pred_sensi.backward()

            self.optimizer.step()
            self.optimizer_pred.step()

            running_loss += loss_pred_cls.item()
            running_adv_loss += loss_pseudo_pred.item()
            running_MI += loss_pred_sensi.item()

            auc += calculate_auc(F.sigmoid(pred_label).cpu().data.numpy(), targets.cpu().data.numpy())
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})

        running_loss /= no_iter
        running_adv_loss /= no_iter
        running_MI /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: cls loss:{}, adv loss:{}, MI:{}'.format(self.epoch, running_loss, running_adv_loss, running_MI))
        #self.log_result('Train epoch', {'cls loss': running_loss, 'adv loss': running_adv_loss, 'AUC': auc}, self.epoch)
        self.epoch += 1