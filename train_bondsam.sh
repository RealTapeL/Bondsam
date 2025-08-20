python -c "import torch; print('CUDA available:', torch.cuda.is_available())"



python train_bondsam.py \
  --epoch 5 \
  --learning_rate 0.001 \
  --batch_size 8 \
  --image_size 224 \
  --model ViT-B-16 \
  --prompting_depth 3 \
  --prompting_length 2 \
  --prompting_branch VL \
  --prompting_type SD \
  --use_hsf \
  --k_clusters 20 \
  --training_data mvtec \
  --use_fastsam \
  --use_memory_bank \
  --mode few_shot \
  --k_shot 1 \
  --device cuda \
  --save_path ./workspaces/bondsam_exp_820