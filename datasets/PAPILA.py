import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class PAPILA(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(PAPILA, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        with open(path_to_pickles, 'rb') as f: 
            self.tol_images = pickle.load(f)
            
        self.A = self.set_A(sens_name) 
        self.Y = (np.asarray(self.dataframe['Diagnosis'].values) > 0).astype('float')
        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img = Image.fromarray(self.tol_images[idx])
        img = self.transform(img)

        label = torch.FloatTensor([item['Diagnosis']])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                
        return idx, img, label, sensitive  