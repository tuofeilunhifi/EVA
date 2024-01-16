export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /workspace/codebase/EVA/EVA-02/asuka/

MODEL_NAME=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

sz=336
batch_size=64
crop_pct=1.0

EVAL_CKPT=/workspace/models/eva02_Ti_pt_in21k_ft_in1k_p14.pt

DATA_PATH=/workspace/datasets/ImageNet-1k/raw/imagenet1k

# using model w/o ema for evaluation (w/o --use_ema_ckpt_eval)
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# --master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
torchrun --nproc_per_node=4 --nnodes=1 \
run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed