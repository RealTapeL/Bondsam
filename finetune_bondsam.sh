#!/bin/bash

# 引线键合检测微调脚本 (使用FastSAM分割后的少样本数据集)
python finetune_bondsam.py \
  --training_data wirebonding_fewshot \
  --testing_data wirebonding_test \
  --save_fig True \
  --model ViT-L-14 \
  --epoch 20 \
  --learning_rate 0.0001 \
  --batch_size 1 \
  --image_size 518 \
  --prompting_depth 4 \
  --prompting_length 5 \
  --prompting_type SD \
  --prompting_branch VL \
  --use_hsf True \
  --k_clusters 20 \
  --save_path ./workspaces/bondsam_finetune \
  --device cuda \
  --pretrained_model ./workspaces/bondsam_exp1/checkpoints/best.pth