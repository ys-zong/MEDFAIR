#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account your_account
#SBATCH --time=00:30:30

OPTIONS=d:
LONGOPTS=experiment:,dataset_name:,sensitive_name:,output_dim:,num_classes:,batch_size:,cross_testing_model_path:,sens_classes:,backbone:,source_domain:,target_domain:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly

! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

eval set -- "$PARSED"

experiment="baseline"
dataset_name="CXP"
sensitive_name="Age"
wandb_name="default"
output_dim=1
num_classes=1
batch_size=1024
backbone="cusResNet18"
sens_classes=2
cross_testing_model_path=""
source_domain=""
target_domain=""

while true; do
    case "$1" in
        --experiment)
            experiment="$2"
            shift 2
            ;;
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --sensitive_name)
            sensitive_name="$2"
            shift 2
            ;;
        --output_dim)
            output_dim="$2"
            shift 2
            ;;
        --num_classes)
            num_classes="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --cross_testing_model_path)
            cross_testing_model_path="$2"
            shift 2
            ;;
        --sens_classes)
            sens_classes="$2"
            shift 2
            ;;        
        --backbone)
            backbone="$2"
            shift 2
            ;;
        --source_domain)
            source_domain="$2"
            shift 2
            ;;
        --target_domain)
            target_domain="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done


python main.py --experiment $experiment --dataset_name $dataset_name --experiment_name $wandb_name --sensitive_name $sensitive_name --output_dim $output_dim --num_classes $num_classes --batch_size $batch_size --sens_classes $sens_classes --cross_testing --cross_testing_model_path $cross_testing_model_path  --test_mode True --backbone $backbone --source_domain $source_domain --target_domain $target_domain