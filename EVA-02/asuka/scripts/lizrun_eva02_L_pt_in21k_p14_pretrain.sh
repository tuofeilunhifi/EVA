#!/bin/bash

##############################################################################################################################
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

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
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
# export NCCL_ASYNC_ERROR_HANDLING=1
### RDMA Config ###

# 简单镜像依赖添加
## 复杂组件打入镜像，小组件可以单独在启动脚本安装
cd /mnt/pfs-guan-ssai/cv/cjy/envs/
pip install torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt

cd /mnt/pfs-guan-ssai/cv/cjy/envs/
TIME=`date "+%Y%m%d-%H%M%S"`
mkdir temp-${TIME}
cp -r /mnt/pfs-guan-ssai/cv/cjy/codebase/apex temp-${TIME}/apex-${RANK}
cd temp-${TIME}/apex-${RANK}
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install xformers==0.0.22 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

# cp -r /mnt/pfs-guan-ssai/cv/cjy/envs/adt ./
# sudo ./adt --token 1852d6b5acf8a1b50a88ac75c3a95136 mountbos

# cd /mnt/pfs-guan-ssai/cv/cjy/envs/
# pip install xformers-0.0.22.post7+cu118-cp310-cp310-manylinux2014_x86_64.whl

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

MODEL=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init

DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-21k/raw/imagenet21k
VAL_DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=32  # 32(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=1.5e-3
b2=0.98
eps=1e-6

dpr=0.1
ls=0.0

epochs=150
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=1

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/mnt/pfs-guan-ssai/cv/cjy/models/eva_clip_psz14.pt


OUTPUT_DIR=/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/logs/${MODEL}

# python -m torch.distributed.launch --nproc_per_node=8 \
#        	--nnodes=$WORLD_SIZE --node_rank=$RANK \
# 	--master_addr=$MASTER_ADDR --master_port=12355 --use_env \
# 5e-4/4e-4/4e-5
# 0.05/0.05/0.05
# 1.0/0.85/0.75
torchrun --nnodes=${WORLD_SIZE} \
  --nproc_per_node=8 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
  run_eva02_pretraining.py \
        --data_path ${DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL} \
        --teacher_type ${teacher_type} \
        --clip_model ${clip_model} \
        --cache_dir ${cache_dir} \
        --input_size ${input_size} --second_input_size ${input_size} \
        --num_mask_patches ${num_mask_patches} \
        --layer_scale_init_value ${ls} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --opt_betas 0.9 ${b2} \
        --opt_eps ${eps} \
        --drop_path ${dpr} \
        --epochs ${epochs} \
        --mixup ${mixup} \
        --color_jitter ${cj} \
        --warmup_epochs ${wmep} \
        --update_freq ${update_freq} \
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed