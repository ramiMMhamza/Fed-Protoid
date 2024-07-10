
# id 
# ls /tmp
# sg cvlab "mkdir /tmp/tmp"
# export https_proxy="http://129.183.4.13:8080" 
# export http_proxy="http://129.183.4.13:8080" 
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_LAUNCH_BLOCKING=1
EPOCHS=${EPOCHS:-1}
ITERS=${ITERS:-50}
PYTHON=${PYTHON:-"python"}
METHOD="FEDPROTO" # FEDPROTO++
path=$1
mkdir ./log/$path
WORK_DIR="./$path/" 
export PYTHONPATH=./FedProtoid/OpenUnReID/
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
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
./cvlab-federated-phd/OpenUnReID/tools/$METHOD/main_fedl.py ./FedProtoid/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
    --launcher="pytorch" --tcp-port=${PORT}  --exec=2. ${@:2}

for i in {1..800}
do  
   python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   ./cvlab-federated-phd/OpenUnReID/tools/$METHOD/main_fedl.py ./FedProtoid/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
       --launcher="pytorch" --tcp-port=${PORT}  --exec=2. ${@:2}
done

