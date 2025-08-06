#!/bin/bash

# 引线键合检测测试脚本 (使用CPU)
python test_bondsam.py \
  --testing_model image \
  --ckt_path ./workspaces/bondsam_exp1/checkpoints/best.pth \
  --save_fig True \
  --image_path ./test_images/wire_bonding_sample.jpg \
  --class_name wire_bonding \
  --save_name bondsam_result.jpg \
  --model ViT-L-14 \
  --image_size 518 \
  --prompting_depth 4 \
  --prompting_length 5 \
  --prompting_type SD \
  --prompting_branch VL \
  --use_hsf True \
  --k_clusters 20 \
  --device cpu