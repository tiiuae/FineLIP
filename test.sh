#!/bin/bash

# Set variables
# CKPT_PATH="experiments/finelip_baseline/ckpt/baseline_128_epoch_6_finelip-B.pt"
CKPT_PATH="experiments/finelip/ckpt/finelip_128_epoch_4_finelip-B.pt"
TEST_DATA="coco" #urban,coco,flickr, docci 

# Use variables
echo "Running evaluation on $TEST_DATA"
python train/test.py --ckpt_path $CKPT_PATH --test_data $TEST_DATA  --run_finelip 
#--finegrain