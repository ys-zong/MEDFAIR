import torch
import numpy as np
from datasets.BaseDataset import BaseDataset


class RadFusion_EHR(BaseDataset):
    def __init__(self, dataframe, data_df, sens_name, sens_classes, transform):
        super(RadFusion_EHR, self).__init__(dataframe, data_df, sens_name, sens_classes, transform)
        
        self.data_df = data_df
        self.A = self.set_A(sens_name) 
        self.Y = (np.asarray(self.dataframe['label'].values) > 0).astype('float')
        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        ehr = self.data_df[self.data_df['idx']==item['idx']].drop(columns = ['idx']).values.squeeze()
        ehr = torch.FloatTensor(ehr)

        label = torch.FloatTensor([int(item['label'])])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return ehr, label, sensitive, idx