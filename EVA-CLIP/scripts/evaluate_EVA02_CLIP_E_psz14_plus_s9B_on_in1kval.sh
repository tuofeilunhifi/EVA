export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL_NAME=EVA02-CLIP-bigE-14-plus

# PRETRAINED=/mnt/pfs-guan-ssai/cv/cjy/models/EVA02_CLIP_E_psz14_plus_s9B.pt
PRETRAINED=/workspace/codebase/EVA/EVA-CLIP/rei/logs/2024_01_26-15_55_50-model_EVA02-CLIP-bigE-14-plus-lr_0.0005-b_900-j_8-p_amp/checkpoints/pytorch_model.bin
# can set PRETRAINED=eva to automaticaly download and load weights; please check details in pretrained.py
# PRETRAINED=eva_clip

DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k/val

torchrun --nproc_per_node=1 --nnodes=1 training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${PRETRAINED} \
        --force-custom-clip \
        --enable_deepspeed \
        # --language="cn" \