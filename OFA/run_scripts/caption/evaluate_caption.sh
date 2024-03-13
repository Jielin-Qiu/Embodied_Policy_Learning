#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=1
export GPUS_PER_NODE=1

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/caption_data/caption_test.tsv
path=../../checkpoints/checkpoint.best_cider_6.1090.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'


python coco_eval.py ../../results/caption/test_predict_1.json ../../dataset/caption_data/test_caption_coco_format.json
