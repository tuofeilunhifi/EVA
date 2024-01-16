export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/

MODEL_NAME=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

PRETRAIN_CKPT=/mnt/pfs-guan-ssai/cv/cjy/models/eva02_Ti_pt_in21k_p14.pt 

OUTPUT_DIR=/mnt/pfs-guan-ssai/cv/cjy/projects/eva/${MODEL_NAME}

DATA_PATH=/workspace/datasets/ImageNet-1k/raw/imagenet1k


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


# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# --master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
torchrun --nproc_per_node=4 --nnodes=1 \
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