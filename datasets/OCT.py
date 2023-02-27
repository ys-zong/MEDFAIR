import torch
import numpy as np
import os
from datasets.BaseDataset import BaseDataset

class OCT(BaseDataset):
    def __init__(self, dataframe, path_to_images, sens_name, sens_classes, transform):
        super(OCT, self).__init__(dataframe, path_to_images, sens_name, sens_classes, transform)

        self.A = self.set_A(sens_name)  
        self.Y = (np.asarray(self.dataframe['label'].values) > 0).astype('float')
        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img = np.load(os.path.join(self.path_to_images, item["Path"]))
        img = self.transform(img)

        label = torch.FloatTensor([item['label']])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
        
        return img, label, sensitive, idx