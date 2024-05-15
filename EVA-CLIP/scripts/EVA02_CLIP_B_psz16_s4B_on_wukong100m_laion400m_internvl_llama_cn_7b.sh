export CUDA_VISIBLE_DEVICES=2
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL=EVA02-CLIP-B-16-InternVL-LLaMA-CN-7B
PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_11/eva_clip_b_224_e10.bin
PRETRAINED_TEXT='/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth'
PRETRAINED_VISUAL_MODEL=EVA02-CLIP-L-14
PRETRAINED_TEXT_MODEL=other
# MODEL=EVA02-CLIP-S-14-InternVL-LLaMA-CN-7B
# MODEL=EVA02-CLIP-B-16-InternVL-LLaMA-CN-7B
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_B_psz14to16.pt
# PRETRAINED_VISUAL_MODEL=EVA02-L-14
# MODEL=Mobile-CLIP-S2
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/mobileclip/mobileclip_s2.pt
# MODEL=Mobile-CLIP-S1
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/mobileclip/mobileclip_s1.pt
# MODEL=Mobile-CLIP-S0
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/mobileclip/mobileclip_s0.pt
# MODEL=CLIP-B-16-InternVL-LLaMA-CN-7B
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/clip/ViT-B-16.pt
# MODEL=CLIP-L-14-InternVL-LLaMA-CN-7B
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/clip/ViT-L-14.pt
# MODEL=CLIP-L-14-336-InternVL-LLaMA-CN-7B
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/clip/ViT-L-14-336px.pt
# PRETRAINED_TEXT='/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth'
# PRETRAINED_VISUAL_MODEL=openai
# PRETRAINED_TEXT_MODEL=other

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
# WUKONG_100M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/subsets-16
MERGE_500M_DATA_PATH="/mnt/pfs-guan-ssai/cv/cjy/data/cc3m/{00000..00000}.tar;/mnt/pfs-guan-ssai/cv/cjy/data/cc12m/{00000..00000}.tar"
VAL_DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k/val

# python -m torch.distributed.launch --nproc_per_node=8 \
#        	--nnodes=$WORLD_SIZE --node_rank=$RANK \
# 	--master_addr=$MASTER_ADDR --master_port=12355 --use_env \
torchrun --nproc_per_node=1 --nnodes=1 \
    training/main.py \
        --save-frequency 10 \
        --zeroshot-frequency 1 \
        --report-to="tensorboard" \
        --wandb-project-name="eva-clip" \
        --wandb-notes="eva02_clip_B_16" \
        --train-num-samples-list 20000000 20000000\
        --dataset-resampled \
        --train-data-list=${MERGE_500M_DATA_PATH} \
        --dataset-type-list="webdataset;webdataset" \
        --imagenet-val=${VAL_DATA_PATH} \
        --warmup 2000 \
        --batch-size=512 \
        --epochs=200 \
        --lr=5e-4 \
        --visual-lr=2e-4 \
        --text-lr=2e-5 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --visual-ld=0.75 \
        --text-ld=0.75 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=8 \
        --model=${MODEL} \
        --pretrained-image=${PRETRAINED_IMAGE} \
        --pretrained-text=${PRETRAINED_TEXT} \
        --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} \
        --pretrained-text-model=${PRETRAINED_TEXT_MODEL} \
        --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token logit_scale image_encoder.model.head.proj visual.proj \
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
        # --precision="amp_bf16" \