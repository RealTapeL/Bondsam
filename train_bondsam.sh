python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python train_bondsam.py \
  --training_data mvtec \
  --testing_data mvtec \
  --save_fig True \
  --model ViT-B-16 \
  --epoch 10 \
  --learning_rate 0.0001 \
  --batch_size 1 \
  --image_size 224 \
  --prompting_depth 4 \
  --prompting_length 5 \
  --prompting_type SD \
  --prompting_branch VL \
  --use_hsf True \
  --k_clusters 20 \
  --save_path ./workspaces/bondsam_exp1 \
  --device cuda \
  --use_anomaly_attention True \
  --use_enhanced_extractor True