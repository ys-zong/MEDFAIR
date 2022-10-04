import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class MIMIC_CXR(BaseDataset):
    def __init__(self, dataframe, PATH_TO_IMAGES, sens_name, sens_classes, transform):
        super(MIMIC_CXR, self).__init__(dataframe, PATH_TO_IMAGES, sens_name, sens_classes, transform)
        
        with open(PATH_TO_IMAGES, 'rb') as f: 
            self.tol_images = pickle.load(f)
            
        self.A = self.set_A(sens_name) 
        self.Y = (np.asarray(self.dataframe['No Finding'].values) > 0).astype('float')
        self.AY_proportion = None

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img = Image.fromarray(self.tol_images[idx]).convert('RGB')
        img = self.transform(img)

        label = torch.FloatTensor([int(item['No Finding'].astype('float') > 0)])

        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                
        return idx, img, label, sensitive