import os
import subprocess


is_3d = False
if is_3d:
    backbone = 'cusResNet18_3d'
    batch_size = 8
else:
    backbone = 'cusResNet18'
    batch_size = 1024

sensitive_name = 'Age'
source_dataset = 'MIMIC_CXR'
target_dataset = 'CXP'
output_dim = num_classes = 1
sens_classes = 5
#bianry_train_multi_test = -1

methods = ['baseline', 'resampling', 'LAFTR', 'CFair', 'LNL', 'EnD', 'DomainInd', 'GroupDRO', 'ODR', 'SWAD', 'resamplingSWAD', 'SAM']
      
    
model_path = 'your_path/fariness_data/model_records/{datas}/{attr}/{bkbone}/'.format(
        datas = source_dataset, attr = sensitive_name, bkbone = backbone)

for method in methods:
    
    method_model_path = os.path.join(model_path, method)
    
    sens_classes = 5
    bianry_train_multi_test = -1
    if method in ['LAFTR', 'CFair']:
        sens_classes = 2
        bianry_train_multi_test = 5
    print(method)
    method_model_path = os.path.join(model_path, method)

    MAIN_CMD = f"sbatch sweep/test/cross_domain/batch_submit.sh" \
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
               f" --bianry_train_multi_test {bianry_train_multi_test}"\
                
    print('command is ', MAIN_CMD)
    CMD = MAIN_CMD.split(' ')
    process = subprocess.Popen(CMD, stdout=subprocess.PIPE, universal_newlines=True)
    out, err = process.communicate()
    print(out)
        
