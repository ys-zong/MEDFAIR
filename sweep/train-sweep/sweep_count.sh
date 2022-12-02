#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account your_account
#SBATCH --time=4:30:30


OPTIONS=d:
LONGOPTS=sweep_id:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

sweep_id="$2"
echo "$sweep_id"

wandb agent --count 1 $sweep_id

echo "done"
