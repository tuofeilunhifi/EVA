export CUDA_VISIBLE_DEVICES=3,4,5,6
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL=EVA02-CLIP-L-14-InternVL-LLaMA-CN-7B
PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_L_psz14.pt
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
# WUKONG_100M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/subsets-16
MERGE_500M_DATA_PATH="/mnt/pfs-guan-ssai/cv/cjy/data/laion2B-en/{00002..00002}.tar"
VAL_DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k/val

# python -m torch.distributed.launch --nproc_per_node=8 \
#        	--nnodes=$WORLD_SIZE --node_rank=$RANK \
# 	--master_addr=$MASTER_ADDR --master_port=12355 --use_env \
torchrun --nproc_per_node=4 --nnodes=1 \
    training/main.py \
        --save-frequency 10 \
        --zeroshot-frequency 1 \
        --report-to="tensorboard" \
        --wandb-project-name="eva-clip" \
        --wandb-notes="eva02_clip_B_16" \
        --train-num-samples 40000000 \
        --dataset-resampled \
        --train-data=${MERGE_500M_DATA_PATH} \
        --dataset-type="webdataset" \
        --imagenet-val=${VAL_DATA_PATH} \
        --warmup 2000 \
        --batch-size=256 \
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
        # --precision="amp_bf16" \