import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc, calculate_metrics

from importlib import import_module

def standard_train(opt, network, optimizer, loader, _criterion, wandb):
    """Train the model for one epoch"""
    train_loss, auc, no_iter = 0., 0., 0
    for i, (index, images, targets, sensitive_attr) in enumerate(loader):
        images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(opt['device'])
        optimizer.zero_grad()
        outputs, _ = network(images)

        loss = _criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())

        train_loss += loss.item()
        no_iter += 1
        
        if opt['log_freq'] and (i % opt['log_freq'] == 0):
            wandb.log({'Training loss': train_loss / no_iter, 'Training AUC': auc / no_iter})

    auc = 100 * auc / no_iter
    train_loss /= no_iter
    return auc, train_loss


def standard_val(opt, network, loader, _criterion, sens_classes, wandb):
    """Compute model output on validation set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
    val_loss, auc, worst_auc = 0., 0., 0.
    no_iter = 0
    with torch.no_grad():
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)
            loss = _criterion(outputs, targets)
            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()
            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                           targets.cpu().data.numpy())
            
            no_iter += 1
            
            if opt['log_freq'] and (i % opt['log_freq'] == 0):
                wandb.log({'Validation loss': val_loss / no_iter, 'Validation AUC': auc / no_iter})

    auc = 100 * auc / no_iter
    val_loss /= no_iter
    log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    
    return auc, val_loss, log_dict, pred_df


def standard_test(opt, network, loader, _criterion, wandb):
    """Compute model output on testing set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    #tol_features = []

    with torch.no_grad():
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)

            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
    return tol_output, tol_target, tol_sensitive, tol_index


class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label  # .cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes)  # .cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            #print('self.parameter[index].shape, self.updated[index].shape, data.shape')
            #print(self.parameter[index].shape, self.updated[index].shape, data.shape)
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()
