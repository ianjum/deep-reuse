#!/bin/bash
#
# This script performs the following operations:
# 1. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=../cifarnet/model/train
EVAL_DIR=../cifarnet/model/evaluation

# Where the dataset is saved to.
DATASET_DIR=../cifarnet/data

# Run evaluation.
python inference.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --number_hashing_functions=20,10 \
  --neuron_vector_length=5,25
