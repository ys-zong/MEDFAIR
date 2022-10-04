import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class MIMIC_III(BaseDataset):
    def __init__(self, dataframe, text_features, sens_name, sens_classes, transform):
        super(MIMIC_III, self).__init__(dataframe, text_features, sens_name, sens_classes, transform)
        
        self.text_features = text_features
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['label'].values) > 0).astype('float')
        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        t_feature = self.text_features[idx]
        t_feature = torch.FloatTensor(t_feature)

        label = torch.FloatTensor([int(item['label'])])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)

        return idx, t_feature, label, sensitive