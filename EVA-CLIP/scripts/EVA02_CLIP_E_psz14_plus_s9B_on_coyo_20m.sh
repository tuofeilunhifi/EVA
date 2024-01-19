export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL=EVA02-CLIP-bigE-14-plus
# PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/EVA02_CLIP_E_psz14_plus_s9B.pt
# PRETRAINED_TEXT=/mnt/pfs-guan-ssai/cv/cjy/models/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189/CLIP-ViT-bigG-14-laion2B-39B-b160k-merge.bin # ckpt is splited into 2 parts. could merge first then load.
# PRETRAINED_VISUAL_MODEL=EVA02-bigE-14
# PRETRAINED_TEXT_MODEL=OpenCLIP-bigG-14

# can automaticaly download and load pretrained models by follwing 4 lines; please check details in pretrained.py
PRETRAINED_IMAGE=eva02_clip
PRETRAINED_TEXT=laion2b_s39b_b160k
PRETRAINED_VISUAL_MODEL=EVA02-CLIP-bigE-14-plus
PRETRAINED_TEXT_MODEL=OpenCLIP-bigG-14


# Following OpenCLIP, we preprocess data by webdataset. We concat paths of LAION-2B and COYO-700M with `;`.
# MERGE_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar;/path/to/coyo700m_en_data/img_data/{000000..047435}.tar"
# LAION_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar"
COYO_20M_DATA_PATH=/mnt/pfs-guan-ssai/cv/yanghongfu/grit_coyo_20m/grit_coyo_20m_all.json
VAL_DATA_PATH=/workspace/datasets/ImageNet-1k/raw/imagenet1k/val

# python -m torch.distributed.launch --nproc_per_node=8 \
#        	--nnodes=$WORLD_SIZE --node_rank=$RANK \
# 	--master_addr=$MASTER_ADDR --master_port=12355 --use_env \
torchrun --nproc_per_node=8 --nnodes=1 \
    training/main.py \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to="tensorboard" \
        # --wandb-project-name="eva-clip" \
        # --wandb-notes="eva02_clip_E_14" \
        --train-num-samples 40000000 \
        --dataset-resampled \
        --train-data=${COYO_20M_DATA_PATH} \
        --dataset-type="json" \
        --imagenet-val=${VAL_DATA_PATH} \
        --warmup 2000 \
        --batch-size=1000 \
        --epochs=100 \
        --lr=5e-4 \
        --visual-lr=4e-4 \
        --text-lr=4e-5 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --visual-ld=0.9 \
        --text-ld=0.75 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=8 \
        --model=${MODEL} \
        --name='eva-vit-4b-14-text-bigG-x-lamb-patch_drop-18nodes-b144k-laion2b' \
        --pretrained-image=${PRETRAINED_IMAGE} \
        --pretrained-text=${PRETRAINED_TEXT} \
        --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} \
        --pretrained-text-model=${PRETRAINED_TEXT_MODEL} \
        --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale \
        --seed 4096 \
        --gather-with-grad \
        --grad-checkpointing \
        --local-loss \
        --force-custom-clip \
        --force-patch-dropout=0.5 \
        --optimizer="lamb" \
        --zero-stage=1 \
        --enable-deepspeed