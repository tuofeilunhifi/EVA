#!/bin/bash

##############################################################################################################################
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# cd /workspace/codebase/EVA/EVA-02/asuka/

# 相对路径问题
# PATH_ORI=${0%/*}
# WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
# cd ${WORK_PATH}

# 多机多卡RDMA相关配置
### RDMA Config ####
# export NCCL_IB_ADAPTIVE_ROUTING=2
# export NCCL_SOCKET_IFNAME=^eth0
# export NCCL_IB_HCA=^mlx5_0
# export NCCL_IB_HCA=^mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
# export NCCL_SOCKET_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_IB_GID_INDEX=3
### RDMA Config ###

# 简单镜像依赖添加
## 复杂组件打入镜像，小组件可以单独在启动脚本安装
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt

cd /mnt/pfs-guan-ssai/cv/cjy/codebase/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install xformers==0.0.22 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/

# master节点路由转IP地址
# 如 of-gptm4d16b2acc1infer-wenlian-master-0 转成 IP
# sleep 30
# waiting for system init
MASTER_IP=""
if [ "${RANK}" == "0" ];then
  while [[ "$MASTER_IP" == "" ]]
    do
        MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
        # MASTER_IP=127.0.0.1
        sleep 1
    done
else
  ## Convert DNS to IP for torch
  MASTER_IP=`getent hosts ${MASTER_ADDR} | awk '{print $1}'` # Ethernet
fi
###############################################################################################################################

MODEL_NAME=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

PRETRAIN_CKPT=/mnt/pfs-guan-ssai/cv/cjy/models/eva02_Ti_pt_in21k_p14.pt 

OUTPUT_DIR=/mnt/pfs-guan-ssai/cv/cjy/projects/eva/${MODEL_NAME}

DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k


sz=336
batch_size=128  # 128(bsz_per_gpu)*8(#gpus_per_node)*1(#nodes)*1(update_freq)=1024(total_bsz)
update_freq=1

lr=2e-4           
lrd=0.9          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=100
wmep=5
dpr=0.1

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=0

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1

# 分布式多机参数和脚本示例
# 系统启动的多机环境会预设置一批用于分布式训练的相关参数
# MASTER_IP：多机主节点IP地址(已转换的)
# MASTER_PORT：多机主节点PORT地址
# WORLD_SIZE：多机机器节点数，默认每个节点的卡数为8(nproc_per_node),如果框架需要多机卡数，
#             可以通过`expr ${WORLD_SIZE} \* 8`获取
# RANK：当前节点在多机节点中的顺序

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# --master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
# torchrun --nproc_per_node=4 --nnodes=1 \
# run_class_finetuning.py \
torchrun --nnodes=${WORLD_SIZE} \
  --nproc_per_node=8 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
  run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed