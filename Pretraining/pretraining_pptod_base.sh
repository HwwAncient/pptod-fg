#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Parameters.
PROJECT_NAME=pptod-fg
SAVE_ROOT=/data3/hww/program/${PROJECT_NAME}
DATASET_PREFIX_PATH=${SAVE_ROOT}/data/pre-training_corpora/tokenized_pretraining_corpora
SAVE_PATH=${SAVE_ROOT}/checkpoints_v2

# Main run.
python -u pretrain.py \
  --dataset_prefix_path ${DATASET_PREFIX_PATH} \
  --save_path ${SAVE_PATH} \
  --save_ckpt_name t5-base-chinese-cluecorpussmall \
  --model_name uer/t5-base-chinese-cluecorpussmall \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --batch_size_per_gpu 8 \
  --number_of_gpu 4 \
  --gradient_accumulation_steps 4 \
  --max_src_len 512 \
  --max_tgt_len 256 \
  --warmup_steps 5000 \
  --max_save_num 10 \
  --save_steps 3000 \
  --seed 11
