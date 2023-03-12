import os
import subprocess
import itertools
import yaml
import wandb
import argparse


parser = argparse.ArgumentParser(description='Hyperparameter sweep.')

parser.add_argument('--dataset_name', type=str, default='HAM10000', help='dataset name')
parser.add_argument('--sensitive_name', type=str, default='Age', help='dataset name')
parser.add_argument('--total_epochs', type=int, default=20, help='total epochs')
parser.add_argument('--sens_classes', type=int, default=2, help='number of sensitive classes')
parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
parser.add_argument('--val_strategy', type=str, default='worst_auc', help='validation strategy')
parser.add_argument('--is_3d', type=bool, default=False, help='whether 3D dataset')
parser.add_argument('--is_slurm', type=bool, default=True, help='whether using Slurm')
parser.add_argument('--resample_which', type=str, default='class', help='what to resample')

args = parser.parse_args()

is_3d = args.is_3d
if is_3d:
    backbone = 'cusResNet18_3d'
    batch_size = 8
#elif is_tabular:
#    backbone = 'cusMLP'
#    batch_size = 512
else:
    backbone = 'cusResNet18'
    batch_size = 1024
    
    
sensitive_name = args.sensitive_name
dataset_name = args.dataset_name
total_epochs = args.total_epochs
output_dim = num_classes = args.num_classes
val_strategy = args.val_strategy
sens_classes = args.sens_classes
resample_which = args.resample_which

methods = ['baseline', 'resampling', 'LAFTR', 'CFair', 'LNL', 'EnD', 'DomainInd', 'GroupDRO', 'ODR', 'SWAD', 'SAM']
    
    
for method in methods:
    print(method)
    project_name = '{dataset} {meth}'.format(dataset = dataset_name, meth = method)
    wandb.init(project=project_name)

    with open('sweep/train-sweep/sweep_{}.yaml'.format(method)) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        config_dict['name'] = '{dataset} {meth} {sens} multiAttr'.format(dataset = dataset_name, meth = method, sens=sensitive_name)
        
        command_list = config_dict['command']
        command_list += ['--dataset_name', dataset_name]
        command_list += ['--experiment_name', '{meth}_{dataset}_{sens}'.format(meth = method, dataset = dataset_name, sens=sensitive_name)]
        command_list += ['--sensitive_name', sensitive_name]
        command_list += ['--total_epochs', total_epochs]
        command_list += ['--output_dim', num_classes]
        command_list += ['--num_classes', num_classes]
        command_list += ['--batch_size', batch_size]
        command_list += ['--val_strategy', val_strategy]
        command_list += ['--sens_classes', sens_classes]
        
        command_list += ['--resample_which', resample_which]
        if is_3d:
            command_list += ['--is_3d', is_3d]
            command_list += ['--backbone', backbone]
        #elif is_tabular:
        #    command_list += ['--is_tabular', is_3d]
        #    command_list += ['--backbone', backbone]
            
        config_dict['command'] = command_list
        #print(config_dict)
    
    sweep_id = wandb.sweep(config_dict, project=project_name)

    counts = 30
    
    for i in range(counts):
        if args.is_slurm:
            MAIN_CMD = f"sbatch sweep/train-sweep/slurm_sweep_count.sh" \
                       f" --sweep_id {sweep_id}" 
        else:
            MAIN_CMD = f"bash sweep/train-sweep/sweep_count.sh" \
                       f" --sweep_id {sweep_id}" \
        
        print('command is ', MAIN_CMD)
        CMD = MAIN_CMD.split(' ')
        process = subprocess.Popen(CMD, stdout=subprocess.PIPE, universal_newlines=True)
        out, err = process.communicate()
        print(out)
