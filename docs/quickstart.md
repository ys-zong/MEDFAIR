# Quick Start

## Installation
Python >= 3.8+ and Pytorch >=1.10 are required for running the code. Other necessary packages are listed in [`environment.yml`](../environment.yml).

### Installation via conda:
```python
cd MEDFAIR/
conda env create -n medfair_env -f environment.yml
conda activate medfair_env
```

## Dataset Download
Due to the data use agreements, we cannot directly share the download link. Please follow the instructions and download datasets via links from the table below:


| **Dataset**  | **Access**                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------|
| CheXpert     | Original data: https://stanfordmlgroup.github.io/competitions/chexpert/                       |
|              | Demographic data: https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf                                                                           |
| MIMIC-CXR    | https://physionet.org/content/mimic-cxr-jpg/2.0.0/                                            |
| PAPILA       | https://www.nature.com/articles/s41597-022-01388-1#Sec6                                       |
| HAM10000     | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T               |
| OCT          | https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm                                    |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k                                                  |
| COVID-CT-MD  |  https://doi.org/10.6084/m9.figshare.12991592                                                 |
| ADNI 1.5T/3T | https://ida.loni.usc.edu/login.jsp?project=ADNI                                               | 


## Usage

### Dataset Preprocessing
See `notebooks/HAM10000.ipynb` for an simple example of how to preprocess the data into desired format.
Basically, it contains 3 steps:
1. Preprocess metadata.
2. Split to train/val/test set
3. Save images into pickle files (optional -- we usually do this for 2D images instead of 3D images, as data IO is not the bottleneck for training 3D images).

After preprocessing, specify the paths of the metadata and pickle files in `configs/datasets.json`.


### Run a single experiment
```python
python main.py --experiment [experiment] --experiment_name [experiment_name] --dataset_name [dataset_name] \
     --backbone [backbone] --total_epochs [total_epochs] --sensitive_name [sensitive_name] \
     --batch_size [batch_size] --lr [lr] --sens_classes [sens_classes]  --val_strategy [val_strategy] \
     --output_dim [output_dim] --num_classes [num_classes]
```
See `parse_args.py` for more options.

### Run a grid search on a Slurm cluster/Regular Machine
```python
python sweep/train-sweep/sweep_batch.py --is_slurm True/False
```
Set the other arguments as needed.


## Model selection and Results analysis
See `notebooks/results_analysis.ipynb` for a step by step example.

## Tabular data
We also implement these algorithms with a three-layer Multi-Layer Perceptron (MLP) as the backbone to explore the tabular data (This is not introduced in the paper). You can use the tabular mode with the parse argument `cusMLP` and `is_tabular`.