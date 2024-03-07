export CUDA_VISIBLE_DEVICES=6,7
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/

# MODEL=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init
MODEL=eva_clip_l_14
student_pretrained=/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_L_psz14.pt

DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-21k/raw/imagenet21k
VAL_DATA_PATH=/mnt/pfs-guan-ssai/cv/rxd/data/ImageNet-1k/raw/imagenet1k # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=32  # 32(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=1.5e-3
b2=0.98
eps=1e-6

dpr=0.1
ls=0.0

epochs=150
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=1

teacher_type=internvl
clip_model=InternVL_C
cache_dir=/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth


OUTPUT_DIR=/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-02/asuka/logs/${MODEL}


# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# --master_addr=${MASTER_ADDR} --master_port=12345 --use_env 
torchrun --nproc_per_node=2 --nnodes=1 \
        run_eva02_pretraining.py \
        --data_path ${DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL} \
        --teacher_type ${teacher_type} \
        --clip_model ${clip_model} \
        --cache_dir ${cache_dir} \
        --input_size ${input_size} --second_input_size ${input_size} \
        --num_mask_patches ${num_mask_patches} \
        --layer_scale_init_value ${ls} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --opt_betas 0.9 ${b2} \
        --opt_eps ${eps} \
        --drop_path ${dpr} \
        --epochs ${epochs} \
        --mixup ${mixup} \
        --color_jitter ${cj} \
        --warmup_epochs ${wmep} \
        --update_freq ${update_freq} \
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed \
        --student_pretrained ${student_pretrained}