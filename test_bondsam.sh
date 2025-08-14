
python test_bondsam.py \
  --testing_model image \
  --ckt_path /home/ps/few-shot-research/Bondsam/checkpoint.pth_best.pth \
  --save_fig True \
  --image_path /home/ps/few-shot-research/AdaCLIP/test_image/006.jpg \
  --class_name wire_bonding \
  --save_name /home/ps/few-shot-research/Bondsam/bondsam_result.jpg \
  --model ViT-B-16 \
  --image_size 224 \
  --prompting_depth 0 \
  --prompting_length 0 \
  --prompting_type '' \
  --prompting_branch '' \
  --use_hsf True \
  --k_clusters 20 \
  --device cuda