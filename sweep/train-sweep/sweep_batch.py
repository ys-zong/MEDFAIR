import os
import subprocess
import itertools
import yaml
import json
import wandb

is_3d = False
is_tabular = True
if is_3d:
    backbone = 'cusResNet18_3d'
    batch_size = 8
elif is_tabular:
    backbone = 'cusMLP'
    batch_size = 512
else:
    batch_size = 1024
    
    
sensitive_name = 'Age'
dataset_name = 'PAPILA'
total_epochs = 20
output_dim = num_classes = 1
val_strategy = 'worst_auc'
sens_classes = 2
bianry_train_multi_test = -1
resample_which = 'class'

methods = ['baseline', 'resampling', 'LAFTR', 'CFair', 'LNL', 'EnD', 'DomainInd', 'GroupDRO', 'ODR', 'SWAD', 'SAM']
    
    
for method in methods:
    print(method)
    project_name = '{dataset} {meth}'.format(dataset = dataset_name, meth = method)
    wandb.init(project=project_name)
    
    bianry_train_multi_test = -1
    sens_classes = 5
    if method in ['LAFTR', 'CFair']:
        bianry_train_multi_test = 5
        sens_classes = 2

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
        
        command_list += ['--bianry_train_multi_test', bianry_train_multi_test]
        command_list += ['--resample_which', resample_which]
        if is_3d:
            command_list += ['--is_3d', is_3d]
            command_list += ['--backbone', backbone]
        elif is_tabular:
            command_list += ['--is_tabular', is_3d]
            command_list += ['--backbone', backbone]
            
        config_dict['command'] = command_list
        #print(config_dict)
    
    sweep_id = wandb.sweep(config_dict, project=project_name)

    counts = 30
    
    for i in range(counts):
               
        MAIN_CMD = f"sbatch sweep/train-sweep/sweep_count.sh" \
                   f" --sweep_id {sweep_id}" \
        
        print('command is ', MAIN_CMD)
        CMD = MAIN_CMD.split(' ')
        process = subprocess.Popen(CMD, stdout=subprocess.PIPE, universal_newlines=True)
        out, err = process.communicate()
        print(out)
