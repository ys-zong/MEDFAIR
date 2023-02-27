import os
import subprocess
import argparse


parser = argparse.ArgumentParser(description='Hyperparameter sweep.')

parser.add_argument('--source_dataset', type=str, default='CXP', help='source dataset name')
parser.add_argument('--target_dataset', type=str, default='MIMIC_CXR', help='target dataset name')
parser.add_argument('--sensitive_name', type=str, default='Age', help='dataset name')
parser.add_argument('--sens_classes', type=int, default=2, help='number of sensitive classes')
parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
parser.add_argument('--is_3d', type=bool, default=False, help='whether 3D dataset')
parser.add_argument('--is_slurm', type=bool, default=True, help='whether using Slurm')

args = parser.parse_args()


if args.is_3d:
    backbone = 'cusResNet18_3d'
    batch_size = 8
else:
    backbone = 'cusResNet18'
    batch_size = 1024

sensitive_name = args.sensitive_name
source_dataset = args.source_dataset
target_dataset = args.target_dataset
output_dim = num_classes = args.num_classes
sens_classes = args.sens_classes

methods = ['baseline', 'resampling', 'LAFTR', 'CFair', 'LNL', 'EnD', 'DomainInd', 'GroupDRO', 'ODR', 'SWAD', 'resamplingSWAD', 'SAM']
      
    
model_path = 'your_path/fariness_data/model_records/{datas}/{attr}/{bkbone}/'.format(
        datas = source_dataset, attr = sensitive_name, bkbone = backbone)

for method in methods:
    
    method_model_path = os.path.join(model_path, method)
    if args.is_slurm:
        MAIN_CMD = f"sbatch sweep/test/cross_domain/slurm_batch_submit.sh" \
                   f" --experiment {method}"\
                   f" --dataset_name {target_dataset}"\
                   f" --sensitive_name {sensitive_name}"\
                   f" --output_dim {output_dim}"\
                   f" --num_classes {num_classes}"\
                   f" --batch_size {batch_size}"\
                   f" --cross_testing_model_path {method_model_path}"\
                   f" --sens_classes {sens_classes}"\
                   f" --backbone {backbone}"\
                   f" --source_domain {source_dataset}"\
                   f" --target_domain {target_dataset}"
    else:
        MAIN_CMD = f"bash sweep/test/cross_domain/batch_submit.sh" \
                   f" --experiment {method}"\
                   f" --dataset_name {target_dataset}"\
                   f" --sensitive_name {sensitive_name}"\
                   f" --output_dim {output_dim}"\
                   f" --num_classes {num_classes}"\
                   f" --batch_size {batch_size}"\
                   f" --cross_testing_model_path {method_model_path}"\
                   f" --sens_classes {sens_classes}"\
                   f" --backbone {backbone}"\
                   f" --source_domain {source_dataset}"\
                   f" --target_domain {target_dataset}"\
                
    print('command is ', MAIN_CMD)
    CMD = MAIN_CMD.split(' ')
    process = subprocess.Popen(CMD, stdout=subprocess.PIPE, universal_newlines=True)
    out, err = process.communicate()
    print(out)
        
