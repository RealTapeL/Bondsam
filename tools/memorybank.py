import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class MemoryBank:
    """
    Memory Bank implementation for few-shot anomaly detection
    Based on VAND-APRIL-GAN implementation
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.memory_features = {}
        
    def build_memory_bank(self, model, dataloader, obj_list, k_shot=10):
        """
        Build Memory Bank to store normal sample features
        """
        mem_features = {}
        model.eval()
        
        # 为类别处理添加进度条
        obj_pbar = tqdm(obj_list, desc="Building Memory Bank", leave=False)
        
        with torch.no_grad():
            for obj_idx, obj in enumerate(obj_pbar):
                obj_pbar.set_description(f"Processing class: {obj}")
                features = []
                count = 0
                
                # 为样本处理添加进度条
                sample_pbar = tqdm(dataloader, desc=f"Samples for {obj}", leave=False, total=min(k_shot, len(dataloader)))
                
                for item_idx, items in enumerate(sample_pbar):
                    if count >= k_shot:
                        break
                        
                    if items['cls_name'][0] == obj:
                        sample_pbar.set_description(f"Processing sample {count+1}/{k_shot} for {obj}")
                        image = items['img'].to(self.device)
                        fastsam_mask = items.get('fastsam_mask', None)
                        if fastsam_mask is not None:
                            fastsam_mask = fastsam_mask.to(self.device)
                        
                        try:
                            image_features, patch_tokens, _ = model.clip_model.extract_feat(
                                image, [obj], fastsam_mask=fastsam_mask
                            )
                            
                            if isinstance(patch_tokens, list):
                                # Only keep necessary features to avoid excessive memory usage
                                processed_tokens = []
                                for p in patch_tokens:
                                    if p.dim() == 3:
                                        token = p[0, 1:, :].to(self.device)
                                    else:
                                        token = p[0].view(p.shape[1], -1).permute(1, 0).contiguous().to(self.device)
                                    processed_tokens.append(token)
                                patch_tokens = processed_tokens
                            else:
                                if patch_tokens.dim() == 3:
                                    patch_tokens = patch_tokens[0, 1:, :].to(self.device)
                                else:
                                    patch_tokens = patch_tokens[0].view(patch_tokens.shape[1], -1).permute(1, 0).contiguous().to(self.device)
                            
                            features.append(patch_tokens)
                            count += 1
                            sample_pbar.set_postfix({"Status": "Success"})
                            
                            # Clean up unnecessary variables promptly
                            del image, image_features
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as e:
                            sample_pbar.set_postfix({"Status": f"Error: {str(e)[:20]}"})
                            continue
                        finally:
                            sample_pbar.update(1)
                
                if features:
                    if isinstance(features[0], list):
                        mem_features[obj] = [torch.cat(
                            [features[j][i] for j in range(len(features))], dim=0
                        ).to(self.device) for i in range(len(features[0]))]
                    else:
                        mem_features[obj] = torch.cat(features, dim=0).to(self.device)
                else:
                    pass
                        
        self.memory_features = mem_features
        return mem_features

    def calculate_anomaly_score_with_memory(self, patch_tokens, obj_name):
        """
        Calculate anomaly score using Memory Bank (GPU optimized version)
        """
        if obj_name not in self.memory_features:
            return None
            
        memory_features = self.memory_features[obj_name]
        
        # Process patch tokens, ensuring they are on the correct device
        if isinstance(patch_tokens, list):
            # Multi-layer feature case
            anomaly_maps_few_shot = []
            for idx, p in enumerate(patch_tokens):
                # Ensure on the correct device
                p = p.to(self.device)
                mem_feat = memory_features[idx].to(self.device)
                
                if p.dim() == 3:
                    p = p[0, 1:, :]
                else:
                    p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                
                # Use batch computation for efficiency
                # Calculate cosine similarity
                p_norm = F.normalize(p, dim=-1)
                mem_norm = F.normalize(mem_feat, dim=-1)
                
                cos_sim = torch.mm(p_norm, mem_norm.t())
                # Convert to anomaly score
                anomaly_map = 1 - cos_sim.max(dim=1)[0]
                anomaly_maps_few_shot.append(anomaly_map)
                
            # Fuse multi-layer features
            anomaly_score = torch.stack(anomaly_maps_few_shot).mean(dim=0)
        else:
            # Single-layer feature case
            patch_tokens = patch_tokens.to(self.device)
            memory_features = memory_features.to(self.device)
            
            if patch_tokens.dim() == 3:
                p = patch_tokens[0, 1:, :]
            else:
                p = patch_tokens[0].view(patch_tokens.shape[1], -1).permute(1, 0).contiguous()
                
            # Use batch computation for efficiency
            # Calculate cosine similarity
            p_norm = F.normalize(p, dim=-1)
            mem_norm = F.normalize(memory_features, dim=-1)
            
            cos_sim = torch.mm(p_norm, mem_norm.t())
            # Convert to anomaly score
            anomaly_score = 1 - cos_sim.max(dim=1)[0]
            
        return anomaly_score


class MultiLayerFeatureFusion:
    """
    Multi-layer feature fusion module with improved feature utilization (GPU optimized version)
    """
    def __init__(self, image_size=224, device='cuda'):
        self.image_size = image_size
        self.device = device
        
    def fuse_features(self, patch_tokens, text_features, weights=None):
        """
        Fuse multi-layer features for anomaly detection
        """
        # Ensure tensors are on the correct device
        patch_tokens = [p.to(self.device) for p in patch_tokens]
        text_features = text_features.to(self.device)
        
        anomaly_maps = []
        
        # Use default weights if not provided
        if weights is None:
            weights = [1.0 / len(patch_tokens)] * len(patch_tokens)
            
        for layer_idx, patch_token in enumerate(patch_tokens):
            # Compute vision-text similarity
            anomaly_map = (100.0 * patch_token @ text_features)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            
            # Adjust size to match image size
            if H * H != L:
                H = int(np.ceil(np.sqrt(L)))
                if H * H > L:
                    padding_size = H * H - L
                    padding = torch.zeros(B, padding_size, C, dtype=anomaly_map.dtype, device=self.device)
                    anomaly_map = torch.cat([anomaly_map, padding], dim=1)
                else:
                    anomaly_map = anomaly_map[:, :H*H, :]
            
            anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
            anomaly_map = F.interpolate(anomaly_map, size=self.image_size, mode='bilinear', align_corners=True)
            anomaly_maps.append(anomaly_map * weights[layer_idx])
            
        # Fuse anomaly maps from all layers
        fused_anomaly_map = torch.stack(anomaly_maps, dim=0).sum(dim=0)
        return fused_anomaly_map, anomaly_maps


class AnomalyClassificationAndSegmentation:
    """
    Anomaly classification and segmentation combination module (GPU optimized version)
    """
    def __init__(self, alpha=0.5, device='cuda'):
        self.alpha = alpha  # Weight for combining classification and segmentation results
        self.device = device
        
    def combine_classification_and_segmentation(self, image_features, patch_tokens, text_features, memory_bank=None, obj_name=None):
        """
        Combine image-level classification and pixel-level segmentation results
        """
        # Ensure tensors are on the correct device
        image_features = image_features.to(self.device)
        patch_tokens = [p.to(self.device) for p in patch_tokens] if isinstance(patch_tokens, list) else patch_tokens.to(self.device)
        text_features = text_features.to(self.device)
        
        # Pixel-level anomaly detection
        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            
            # Adjust size to match image size
            if H * H != L:
                H = int(np.ceil(np.sqrt(L)))
                if H * H > L:
                    padding_size = H * H - L
                    padding = torch.zeros(B, padding_size, C, dtype=anomaly_map.dtype, device=self.device)
                    anomaly_map = torch.cat([anomaly_map, padding], dim=1)
                else:
                    anomaly_map = anomaly_map[:, :H*H, :]
            
            anomaly_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
            anomaly_map = F.interpolate(anomaly_map, size=224, mode='bilinear', align_corners=True)
            anomaly_maps.append(anomaly_map)
            
        # Image-level classification
        anomaly_score = (100.0 * image_features.unsqueeze(1) @ text_features)
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)
        
        # Combine with memory bank features if available
        if memory_bank and obj_name:
            memory_score = memory_bank.calculate_anomaly_score_with_memory(patch_tokens, obj_name)
            if memory_score is not None:
                # Combine memory bank score with existing score
                memory_score_normalized = (memory_score - memory_score.min()) / (memory_score.max() - memory_score.min() + 1e-8)
                anomaly_score_combined = self.alpha * anomaly_score[:, 1] + (1 - self.alpha) * memory_score_normalized.mean()
                return anomaly_maps, anomaly_score_combined
        
        return anomaly_maps, anomaly_score[:, 1]