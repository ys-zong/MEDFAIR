import os
import torch
import torch.nn.functional as F
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.baseline import baseline


class DomainInd(baseline):
    def __init__(self, opt, wandb):
        super(DomainInd, self).__init__(opt, wandb)

        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)

    def _criterion_domain(self, output, target, sensitive_attr):
        domain_label = sensitive_attr.long() #.reshape(-1, 1)
        class_num = output.shape[1] // self.sens_classes
        preds = []
        for i in range(domain_label.shape[0]):
            preds.append(output[i, domain_label[i] * class_num: (domain_label[i]+1) *class_num])
        preds = torch.stack(preds)
        loss = F.binary_cross_entropy_with_logits(preds, target)
        return loss    
    
    def inference_sum_prob(self, output):
        """Inference method: sum the probability from multiple domains"""
        #predict_prob = torch.sigmoid(output)
        predict_prob = output
        class_num = predict_prob.shape[1] // self.sens_classes
        predict_prob_sum = []
        for i in range(self.sens_classes):
            predict_prob_sum.append(predict_prob[:, i * class_num: (i+1) * class_num])
        predict_prob_sum = torch.stack(predict_prob_sum).sum(0)
        predict_prob_sum = torch.sigmoid(predict_prob_sum)
        return predict_prob_sum
    
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        train_loss, auc, no_iter = 0, 0., 0
        
        for i, (index, images, targets, sensitive_attr) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.network.forward(images)
    
            loss = self._criterion_domain(outputs, targets, sensitive_attr)
            loss.backward()
            self.optimizer.step()
    
            outputs = self.inference_sum_prob(outputs)
            auc += calculate_auc(outputs.cpu().data.numpy(),
                                               targets.cpu().data.numpy())
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
        val_loss, auc = 0., 0.
        no_iter = 0
        with torch.no_grad():
            for i, (index, images, targets, sensitive_attr) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                loss = self._criterion_domain(outputs, targets, sensitive_attr)
                val_loss += loss.item()
                outputs = self.inference_sum_prob(outputs)
                
                tol_output += outputs.flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                    
                auc += calculate_auc(outputs.cpu().data.numpy(),
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
        """Compute model output on testing set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for i, (index, images, targets, sensitive_attr) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                outputs = self.inference_sum_prob(outputs)
    
                tol_output += outputs.flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()

        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, self.experiment + '_pred.csv'), index = False)
        
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
            
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')

        return log_dict
