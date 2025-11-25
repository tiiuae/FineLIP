#!/bin/bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES=0,1,2,3
 
GPUs=4
GPUs_per_node=4
 
WORLD_SIZE=$(($GPUs))

# for baseline, run:
# torchrun --nproc_per_node=$GPUs_per_node \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_port=25678 \
#     train/train.py \
#         --exp_name="baseline" \
#         --run_baseline \
#         --enable_wandb 


torchrun --nproc_per_node=$GPUs_per_node \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=25678 \
    train/train.py \
        --exp_name="finelip" \
        --enable_wandb 