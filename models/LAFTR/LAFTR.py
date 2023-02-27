import torch
import torch.nn.functional as F
from models.LAFTR.model import LaftrNet, LaftrNet3D, LaftrNet_MLP
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.utils import standard_val, standard_test
from models.basenet import BaseNet


class LAFTR(BaseNet):
    def __init__(self, opt, wandb):
        super(LAFTR, self).__init__(opt, wandb)

        self.model_var = opt['model_var']
        self.test_classes = opt['sens_classes']
        self.sens_classes = 2

        self.set_network(opt)
        self.set_optimizer(opt)

        self.aud_steps = opt['aud_steps']
        self.class_coeff = opt['class_coeff']
        self.fair_coeff = opt['fair_coeff']

    def set_network(self, opt):
        """Define the network"""
        if self.is_3d:
            self.network = LaftrNet3D(backbone = self.backbone, num_classes=self.num_classes, adversary_size=128, pretrained = self.pretrained, device=self.device, model_var=self.model_var).to(self.device)
        elif self.is_tabular:
            self.network = LaftrNet_MLP(backbone = self.backbone, num_classes=self.num_classes, adversary_size=128, device=self.device, model_var=self.model_var, in_features=self.in_features, hidden_features=1024).to(self.device)
        else:
            self.network = LaftrNet(backbone = self.backbone, num_classes=self.num_classes, adversary_size=128, pretrained = self.pretrained, device=self.device, model_var=self.model_var).to(self.device)
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
            params=self.network.net.parameters(),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )
        self.optimizer_disc = optimizer_setting['optimizer'](
            params=filter(lambda p: p.requires_grad, self.network.discriminator.parameters()),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )

    def get_AYweights(self, data):
        A_weights, Y_weights, AY_weights = (
            data.get_A_proportions(),
            data.get_Y_proportions(),
            data.get_AY_proportions(),
        )
        return A_weights, Y_weights, AY_weights

    def l1_loss(self, y, y_logits):
        """Returns l1 loss"""
        y_hat = torch.sigmoid(y_logits)
        return torch.squeeze(torch.abs(y - y_hat))

    def get_weighted_aud_loss(self, L, X, Y, A, A_wts, Y_wts, AY_wts):
        """Returns weighted discriminator loss"""
        Y = Y[:, 0]
        if self.model_var == "laftr-dp":
            A0_wt = A_wts[0]
            A1_wt = A_wts[1]
            wts = A0_wt * (1 - A) + A1_wt * A
            wtd_L = L * torch.squeeze(wts)
        elif (
                self.model_var == "laftr-eqodd"
                or self.model_var == "laftr-eqopp0"
                or self.model_var == "laftr-eqopp1"
        ):
            A0_Y0_wt = AY_wts[0][0]
            A0_Y1_wt = AY_wts[0][1]
            A1_Y0_wt = AY_wts[1][0]
            A1_Y1_wt = AY_wts[1][1]

            if self.model_var == "laftr-eqodd":
                wts = (
                        A0_Y0_wt * (1 - A) * (1 - Y)
                        + A0_Y1_wt * (1 - A) * (Y)
                        + A1_Y0_wt * (A) * (1 - Y)
                        + A1_Y1_wt * (A) * (Y)
                )
            elif self.model_var == "laftr-eqopp0":
                wts = A0_Y0_wt * (1 - A) * (1 - Y) + A1_Y0_wt * (A) * (1 - Y)
            elif self.model_var == "laftr-eqopp1":
                wts = A0_Y1_wt * (1 - A) * (Y) + A1_Y1_wt * (A) * (Y)

            wtd_L = L * torch.squeeze(wts)
        else:
            raise Exception("Wrong model name")
            exit(0)

        return wtd_L

    def _train(self, loader):
        """Train the model for one epoch"""
        A_weights, Y_weights, AY_weights = self.get_AYweights(self.train_data)

        self.network.train()

        running_loss = 0.
        running_adv_loss = 0.
        auc = 0.
        no_iter = 0
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                self.device)
            self.optimizer.zero_grad()
            Y_logits, A_logits = self.network.forward(images, targets)

            class_loss = self.class_coeff * self._criterion(Y_logits, targets)
            aud_loss = -self.fair_coeff * self.l1_loss(sensitive_attr, A_logits)
            weighted_aud_loss = self.get_weighted_aud_loss(aud_loss, images, targets, sensitive_attr, A_weights,
                                                           Y_weights, AY_weights)
            weighted_aud_loss = torch.mean(weighted_aud_loss)
            loss = class_loss + weighted_aud_loss

            torch.autograd.set_detect_anomaly(True)

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.net.parameters(), 5.0)

            for i in range(self.aud_steps):
                if i != self.aud_steps - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.discriminator.parameters(), 5.0)
                self.optimizer_disc.step()
            self.optimizer.step()

            running_loss += loss.item()
            running_adv_loss += weighted_aud_loss.item()

            auc += calculate_auc(F.sigmoid(Y_logits).cpu().data.numpy(),
                                           targets.cpu().data.numpy())

            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})
            

        running_loss /= no_iter
        running_adv_loss /= no_iter

        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: cls loss:{}, adv loss:{}'.format(
            self.epoch, running_loss, running_adv_loss))

        self.epoch += 1
        
    def _val(self, loader):
        
        self.network.eval()
        auc, val_loss, log_dict, pred_df = standard_val(self.opt, self.network, loader, self._criterion, self.test_classes, self.wandb)
        
        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
    
    def _test(self, loader):

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
        return log_dict
