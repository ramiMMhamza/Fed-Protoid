#!/bin/bash
set -x


# id 
ls /data
# sg cvlab "mkdir /tmp/tmp"
export https_proxy="http://129.183.4.13:8080" 
export http_proxy="http://129.183.4.13:8080" 
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_LAUNCH_BLOCKING=1
PYTHON=${PYTHON:-"python"}
METHOD="FEDPROTO" #SpCLPROTO (Fed-Ta-Proto)FEDPROTO (Fed-So-Proto) "SpCL" strong_baseline2 MMT2(MMT2_m: mean on KD_loss, MMT2_new: mean_net is the teacher) SpCL_new(mean_net addet and cosidered as teacher)
WORK_DIR="/out/" 
PY_ARGS=${@:3}
export PYTHONPATH=/exp/OpenUnReID/
GPUS=${GPUS:-2}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
# echo $@
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# /exp/OpenUnReID/tools/$METHOD/main_fedl.py /exp/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
#     --launcher="pytorch" --tcp-port=${PORT}  --exec=1. $@ ||
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# /exp/OpenUnReID/tools/$METHOD/main_debug_fedl.py /exp/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
#     --launcher="pytorch" --tcp-port=${PORT}  --exec=1. $@

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
/exp/OpenUnReID/tools/$METHOD/main_fedl.py /exp/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
    --launcher="pytorch" --tcp-port=${PORT}  --exec=1. $@ 
for i in {1..800}
do  
    python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    /exp/OpenUnReID/tools/$METHOD/main_debug_fedl.py /exp/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
        --launcher="pytorch" --tcp-port=${PORT}  --exec=1. $@ 
done
