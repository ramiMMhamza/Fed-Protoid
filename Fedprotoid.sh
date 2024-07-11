#!/bin/bash

#SBATCH --output=logs/FEDTRANSREID_S_M_LUP_S_10times%j.out
#SBATCH --error=logs/FEDTRANSREID_S_M_LUP_S_10times%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --partition=A40

set -x

cd /home/ids/hrami
pwd

echo $CUDA_VISIBLE_DEVICES
eval "$(conda shell.bash hook)"
conda init bash
conda activate fedreid
echo 'Virtual environment activated'
./run_cmd_train.sh FEDPROTOREID_S_M --epochs_s 1 --iters_s 0 --epochs_t 1 --iters_t 0 --times_itc 10
wait
conda deactivate
echo 'python scripts have finished'


