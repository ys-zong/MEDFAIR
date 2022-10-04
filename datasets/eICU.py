import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class eICU(BaseDataset):
    def __init__(self, dataframe, s_features, sens_name, sens_classes, transform):
        super(eICU, self).__init__(dataframe, s_features, sens_name, sens_classes, transform)
        
        self.s_features = s_features
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['mortality_LABEL'].values) > 0).astype('float')
        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        patient_idx = int(item['patientunitstayid'])
        feature_idx = int(np.where(self.s_features[:, -1]==patient_idx)[0])
        s_feature = self.s_features[feature_idx, :-1]
        s_feature = torch.FloatTensor(s_feature)

        label = torch.FloatTensor([int(item['mortality_LABEL'])])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return idx, s_feature, label, sensitive