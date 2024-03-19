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
pip install pydantic==1.10.9
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

cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

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

MODEL=EVA02-CLIP-L-14-InternVL-LLaMA-CN-7B
PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_L_psz14.pt
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/logs/eva_clip_l_14/eva02_internvit_e20.bin
PRETRAINED_TEXT='/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth'
PRETRAINED_VISUAL_MODEL=EVA02-L-14
PRETRAINED_TEXT_MODEL=other

# can automaticaly download and load pretrained models by follwing 4 lines; please check details in pretrained.py
# PRETRAINED_IMAGE=eva
# PRETRAINED_TEXT=openai
# PRETRAINED_VISUAL_MODEL=EVA02-B-16
# PRETRAINED_TEXT_MODEL=OpenaiCLIP-B-16


# Following OpenCLIP, we preprocess data by webdataset. We concat paths of LAION-2B and COYO-700M with `;`.
# MERGE_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar;/path/to/coyo700m_en_data/img_data/{000000..047435}.tar"
# LAION_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar"
# WUKONG_100M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/wukong-all.json
# WUKONG_100M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/subsets-16/wukong-100m-part-0.json
# WUKONG_100M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/original/putput_wukong_100m_0.json
# MERGE_500M_DATA_PATH="/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/recipe/{00000..00127}.tar"
MERGE_500M_DATA_PATH="/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/recipe_1200m/{00000..00127}.tar"
# MERGE_500M_DATA_PATH="/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/recipe_laion_200m/{00000..00017}.tar;/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/wukong/wukong-100m-part-{0..15}.tar"
VAL_DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k/val

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
    training/main.py \
        --save-frequency 10 \
        --zeroshot-frequency 1 \
        --report-to="wandb, tensorboard" \
        --wandb-project-name="eva-clip" \
        --wandb-notes="eva02_clip_L_14" \
        --train-num-samples 40000000 \
        --dataset-resampled \
        --train-data=${MERGE_500M_DATA_PATH} \
        --dataset-type="webdataset" \
        --imagenet-val=${VAL_DATA_PATH} \
        --warmup 2000 \
        --batch-size=1024 \
        --epochs=100 \
        --lr=5e-4 \
        --visual-lr=4e-4 \
        --text-lr=4e-5 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --visual-ld=0.85 \
        --text-ld=0.75 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=8 \
        --model=${MODEL} \
        --pretrained-image=${PRETRAINED_IMAGE} \
        --pretrained-text=${PRETRAINED_TEXT} \
        --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} \
        --pretrained-text-model=${PRETRAINED_TEXT_MODEL} \
        --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token logit_scale \
        --seed 4096 \
        --gather-with-grad \
        --grad-checkpointing \
        --local-loss \
        --force-custom-clip \
        --force-patch-dropout=0 \
        --optimizer="lamb" \
        --zero-stage=1 \
        --enable-deepspeed \
        --language="cn" \
        --lock-text \