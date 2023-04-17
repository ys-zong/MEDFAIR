# Not Enough? Customize your own experiments.
In our benchmarking framework, you can easily add different datasets, network architectures, debiasing algorithms, and evaluation metrics for your own experiments.

## Customize Dataset
You can easily add any dataset you need following the three steps below.

### STEP 1. Configure dataset
Preprocess the dataset and image files in a way similar to `notebooks/HAM10000-example.ipynb`.

### STEP 2. Implement the Dataset Class
We write the dataset class inheriting the regular Pytorch Dataset ([official tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)). We provide a base dataset class in `datasets/BaseDataset.py`. In `datasets` folder, create a new script named after your dataset (e.g. `DatasetX.py`), and name the new dataset class with the same name as the script (i.e. `class DatasetX`). An example script is given below. The input paths need to be specified in `configs/datasets.json`. The comments in the code block below may be helpful.

```python
import torch
import pickle
import numpy as np
from PIL import Image
from datasets.BaseDataset import BaseDataset

class DatasetX(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(DatasetX, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        """
            Dataset class for customized dataset
            
            Arguments:
            dataframe: the metadata in pandas dataframe format.
            path_to_pickles: path to the pickle file containing images.
            sens_name: which sensitive attribute to use, e.g., Sex.
            sens_classes: number of sensitive classes.
            transform: whether conduct data transform to the images or not.
            
            Returns:
            index, image, label, and sensitive attribute.
        """
        
        # load the pickle file containing all images
        with open(path_to_pickles, 'rb') as f: 
            self.tol_images = pickle.load(f)
            
        self.A = set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['binaryLabel'].values) > 0).astype('float')
        self.AY_proportion = None

    def __getitem__(self, idx):
        # get the item based on the index
        item = self.dataframe.iloc[idx]

        # get the image from the pickle file
        img = Image.fromarray(self.tol_images[idx])
        # uncomment the line to load the image directly below if you don't want to use pickle file.
        # Note, the `path_to_images` variable needs to be modified accordingly.
        # img = Image.open(path_to_images[idx])

        # apply image transform/augmentation
        img = self.transform(img)

        label = torch.FloatTensor([int(item['binaryLabel'])])
        
        # get sensitive attributes in numerical values
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return img, label, sensitive, idx
```

You can also refer to other dataset classes we wrote in the `datasets` folder.

### STEP 3. Register the dataset
- Add the dataset name in the choices of `dataset_name` argument in `parse_args.py`.
- Import the dataset class in `datasets/__init__.py`.
- Make sure the paths to the dataset is written to the `configs/datasets.json`.

Now, you can use your own dataset for training!

## Customize Network Architectures
You can add more network architectures (CNN-based) to the framework easily. Transformer models can also be incorporated yet requires some other modifications. 

### STEP 1. Implement the Network Class
You can incorporate any 2D model in `models/basemodels.py` and 3D model in `models/basemodels_3d.py`. We use the backbone provided by torchvision model zoo, but you can also implement your customized network structures. We use the `create_feature_extractor` function to extract the intermediate feature map. An example is given below:

```python
class cusResNet18(nn.Module):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet18, self).__init__()
        # load the model backbone
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        # change the output neuron of the fc layer
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
        self.avgpool = resnet.avgpool
        
        # specific the feature layer you want to extract
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})

    def forward(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()
```

### STEP 2. Register the Network
- Add the network backbone name in the choices of `backbone` argument in `parse_args.py`.

## Customize Debiasing Algorithms

### STEP 1. Implement the Algorithm Class
For debiasing algorithms, we provide a base algorithm class in `models/basenet.py`, which includes necessary initializations, training/testing loop, and evaluations. To add a new algorithm, you should first create a new folder named after your algorithm under `models` folder (e.g. `AlgorithmX`), and create a `__init__.py` file and a `AlgorithmX.py` file. In `AlgorithmX.py`, name the new algorithm class with the same name as the script (i.e. `class AlgorithmX`). Then, follow the three steps below to implement your algorithm:

1. Configure Hyper-parameters.
   
   You can add options for the algorithm-specific hyper-parameter in `parse_args.py`. For example for EnD method, we set two hyper-parameters `alpha` and `beta`:
   ```python
   # EnD
   parser.add_argument('--alpha', type=float, default=0.1, help='weighting parameters alpha for EnD method')
   parser.add_argument('--beta', type=float, default=0.1, help='weighting parameters beta for EnD method')
   ```

2. Network and other utils.
   
   - If your method does not require customized network architecture, you can use the regular networks as in `models/basemodels.py` for 2D networks and `models/basemodels_3d.py` for 3D networks. 
   - If you need to modify the network architecture, you can create a file `models/AlgorithmX/model.py` and implement the network class there with a `forward` function for training, and a `inference` function for testing. Example implementation can be referred to `models/LAFTR/model.py`, etc. 
   - If you need other functions for training, you can create a file `models/AlgorithmX/utils.py` and implement it there.
   - Import modules you need in `models/AlgorithmX/__init__.py`.

3. Training loop
   - If the train/val/test procedure is the regular loop like that in `models/baseline.py`, and does not require other loss functions, backward propagation, other outputs, etc. (check `standard_train`, `standard_val`, `standard_val` functions in `models/utils.py`), you do not need further modifications.
   - If you want to write your own training loop, you can override a new `_train` function within the `AlgorithmX` class. You can have a look at the implementation of other algorithms in `models` folder for reference. Also, override the `_val` and `_test` function if needed.


### STEP 2. Register the Algorithm
- Add the algorithm name (`AlgorithmX`) in the choices of `experiment` argument in `parse_args.py`.
- Import the algorithm class in `models/__init__.py`.

Now, you can use your own algorithm for training!

## Customize Evaluation Metrics
Currently, we implement the evaluation metrics in `utils/evaluation.py` and record all of them in `calculate_metrics` function. You can implement the evaluation metrics in this file and then add it in the `calculate_metrics` function.