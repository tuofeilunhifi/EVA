export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/rei/

MODEL=EVA02-CLIP-bigE-14
PRETRAINED_IMAGE=/mnt/pfs-guan-ssai/cv/cjy/models/EVA02_E_psz14.pt
PRETRAINED_TEXT=/mnt/pfs-guan-ssai/cv/cjy/models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin # ckpt is splited into 2 parts. could merge first then load.
PRETRAINED_VISUAL_MODEL=EVA02-bigE-14
PRETRAINED_TEXT_MODEL=OpenCLIP-H-14

# can automaticaly download and load pretrained models by follwing 4 lines; please check details in pretrained.py
# PRETRAINED_IMAGE=eva
# PRETRAINED_TEXT=laion2b_s32b_b79k
# PRETRAINED_VISUAL_MODEL=EVA02-bigE-14
# PRETRAINED_TEXT_MODEL=OpenCLIP-H-14


# Following OpenCLIP, we preprocess data by webdataset. We concat paths of LAION-2B and COYO-700M with `;`.
# MERGE_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar;/path/to/coyo700m_en_data/img_data/{000000..047435}.tar"
# LAION_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar"
COYO_20M_DATA_PATH=/mnt/pfs-guan-ssai/cv/cjy/data/coyo/grit_coyo_20m_all_2.json
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
        --wandb-notes="eva02_clip_E_14" \
        --train-num-samples 40000000 \
        --dataset-resampled \
        --train-data=${COYO_20M_DATA_PATH} \
        --dataset-type="json" \
        --imagenet-val=${VAL_DATA_PATH} \
        --warmup 2000 \
        --batch-size=800 \
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