#!/bin/bash

# 引线键合检测训练脚本 (使用MVTec数据集)
python train_bondsam.py \
  --training_data mvtec \
  --testing_data mvtec \
  --save_fig True \
  --model ViT-B-16 \
  --epoch 5 \
  --learning_rate 0.001 \
  --batch_size 1 \
  --image_size 224 \
  --prompting_depth 4 \
  --prompting_length 5 \
  --prompting_type SD \
  --prompting_branch VL \
  --use_hsf True \
  --k_clusters 20 \
  --save_path ./workspaces/bondsam_exp1 \
  --device cuda