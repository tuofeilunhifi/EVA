export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL_NAME=EVA02-CLIP-bigE-14-plus

PRETRAINED=/mnt/pfs-guan-ssai/cv/cjy/models/EVA02_CLIP_E_psz14_plus_s9B.pt
# can set PRETRAINED=eva to automaticaly download and load weights; please check details in pretrained.py
# PRETRAINED=eva_clip

DATA_PATH=/workspace/datasets/ImageNet-1k/raw/imagenet1k/val

torchrun --nproc_per_node=1 --nnodes=1 training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${PRETRAINED} \
        --force-custom-clip \
        --enable_deepspeed