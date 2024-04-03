export CUDA_VISIBLE_DEVICES=7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL_NAME=EVA02-CLIP-L-14-InternVL-LLaMA-CN-7B

# PRETRAINED=/mnt/pfs-guan-ssai/cv/cjy/models/EVA02_CLIP_E_psz14_plus_s9B.pt
PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_01/eva_clip_l_e20.bin
PRETRAINED_TEXT='/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth'
PRETRAINED_VISUAL_MODEL=EVA02-CLIP-L-14
PRETRAINED_TEXT_MODEL=other


DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k/val
FLICKR30K_PATH=/mnt/pfs-guan-ssai/cv/cjy/data/flick30k

torchrun --nproc_per_node=1 --nnodes=1 training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained-image=${PRETRAINED_IMAGE} \
        --pretrained-text=${PRETRAINED_TEXT} \
        --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} \
        --pretrained-text-model=${PRETRAINED_TEXT_MODEL} \
        --force-custom-clip \
        --enable_deepspeed \
        --flickr30k ${FLICKR30K_PATH} \
        --language="cn" \