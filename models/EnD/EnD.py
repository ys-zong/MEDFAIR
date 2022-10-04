import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EnD.model import EnDNet, EnDNet3D,EnDNetMLP
from utils.evaluation import calculate_auc
from models.basenet import BaseNet


class EnD(BaseNet):
    def __init__(self, opt, wandb):
        super(EnD, self).__init__(opt, wandb)

        self.set_network(opt)
        self.set_optimizer(opt)

        self.alpha = opt['alpha']
        self.beta = opt['beta']

    def set_network(self, opt):
        """Define the network"""
        if self.is_3d:
            self.network = EnDNet3D(backbone = self.backbone, n_classes = self.num_classes, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            self.network = EnDNetMLP(backbone = self.backbone, n_classes = self.num_classes, in_features=self.in_features, hidden_features=1024).to(self.device)
        else:
            self.network = EnDNet(backbone = self.backbone, n_classes = self.num_classes, pretrained = self.pretrained).to(self.device)

    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()

        running_loss = 0.
        running_adv_loss = 0.
        auc = 0.
        no_iter = 0
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                self.device)
            self.optimizer.zero_grad()
            outputs, features = self.network.forward(images)

            bce_loss = self._criterion(outputs, targets)
            abs_loss = self.abs_regu(features, targets, sensitive_attr, self.alpha, self.beta)
            loss = bce_loss + abs_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_adv_loss += abs_loss.item()

            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                           targets.cpu().data.numpy())

            no_iter += 1
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})
            
        #self.scheduler.step()

        running_loss /= no_iter
        running_adv_loss /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: cls loss:{}, adv loss:{}'.format(
            self.epoch, running_loss, running_adv_loss))
        
        self.epoch += 1

    def abs_orthogonal_blind(self, output, gram, target_labels, bias_labels):
        # For each discriminatory class, orthogonalize samples

        bias_classes = torch.unique(bias_labels)
        orthogonal_loss = torch.tensor(0.).to(output.device)
        M_tot = 0.

        for bias_class in bias_classes:
            bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)
            bias_mask = torch.tril(torch.mm(bias_mask, torch.transpose(bias_mask, 0, 1)), diagonal=-1)
            M = bias_mask.sum()
            M_tot += M

            if M > 0:
                orthogonal_loss += torch.abs(torch.sum(gram * bias_mask))

        if M_tot > 0:
            orthogonal_loss /= M_tot
        return orthogonal_loss

    def abs_parallel(self, gram, target_labels, bias_labels):
        # For each target class, parallelize samples belonging to
        # different discriminatory classes

        target_classes = torch.unique(target_labels)
        bias_classes = torch.unique(bias_labels)

        parallel_loss = torch.tensor(0.).to(gram.device)
        M_tot = 0.

        for target_class in target_classes:
            class_mask = (target_labels == target_class).type(torch.float).unsqueeze(dim=1)

            for idx, bias_class in enumerate(bias_classes):
                bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)

                for other_bias_class in bias_classes[idx:]:
                    if other_bias_class == bias_class:
                        continue

                    other_bias_mask = (bias_labels == other_bias_class).type(torch.float).unsqueeze(dim=1)
                    mask = torch.tril(
                        torch.mm(class_mask * bias_mask, torch.transpose(class_mask * other_bias_mask, 0, 1)),
                        diagonal=-1)
                    M = mask.sum()
                    M_tot += M

                    if M > 0:
                        parallel_loss -= torch.sum((1.0 + gram) * mask * 0.5)
        if M_tot > 0:
            parallel_loss = 1.0 + (parallel_loss / M_tot)
        return parallel_loss

    def abs_regu(self, feat, target_labels, bias_labels, alpha=1.0, beta=1.0, sum=True):
        D = feat
        if len(D.size()) > 2:
            D = D.view(-1, np.prod((D.size()[1:])))

        gram_matrix = torch.tril(torch.mm(D, torch.transpose(D, 0, 1)), diagonal=-1)
        # not really needed, just for safety for approximate repr
        gram_matrix = torch.clamp(gram_matrix, -1, 1.)

        zero = torch.tensor(0.).to(target_labels.device)
        R_ortho = self.abs_orthogonal_blind(D, gram_matrix, target_labels, bias_labels) if alpha != 0 else zero
        R_parallel = self.abs_parallel(gram_matrix, target_labels, bias_labels) if beta != 0 else zero

        if sum:
            return alpha * R_ortho + beta * R_parallel
        return alpha * R_ortho, beta * R_parallel
