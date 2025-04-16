#!/bin/bash

# Setup environment
export WANDB_API_KEY="your_api_key"
pip install -r requirements.txt  # Ensure all dependencies are installed

# Multi-GPU training command
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gpt2.py \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --num_epochs 3