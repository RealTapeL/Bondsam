import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


class MemoryBank:
    """
    Memory Bank implementation for few-shot anomaly detection
    Based on VAND-APRIL-GAN implementation
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.memory_features = {}
        self.clustered_memory_features = {}
        
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

    def cluster_memory_features(self, n_clusters=32, method='kmeans'):
        """
        Apply clustering to memory features to reduce redundancy and improve efficiency
        """
        clustered_features = {}
        
        for obj_name, features in self.memory_features.items():
            if isinstance(features, list):
                # 多层特征聚类
                clustered_layers = []
                for layer_features in features:
                    if layer_features.shape[0] > n_clusters:
                        if method == 'kmeans':
                            # 使用K-means聚类
                            features_cpu = layer_features.cpu().numpy()
                            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features_cpu)
                            clustered_layer = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
                        elif method == 'adaptive_pooling':
                            # 使用自适应池化
                            # 将特征重塑为可以池化的形式
                            n_features = layer_features.shape[0]
                            if n_features > n_clusters:
                                # 计算池化窗口大小
                                pool_size = n_features // n_clusters
                                # 使用自适应平均池化
                                pooled = F.adaptive_avg_pool1d(
                                    layer_features.permute(1, 0).unsqueeze(0), 
                                    n_clusters
                                ).squeeze(0).permute(1, 0)
                                clustered_layer = pooled
                            else:
                                clustered_layer = layer_features
                        else:
                            # 默认使用简单的随机采样
                            indices = torch.randperm(layer_features.shape[0])[:n_clusters]
                            clustered_layer = layer_features[indices]
                    else:
                        clustered_layer = features
                    
                    clustered_layers.append(clustered_layer)
                clustered_features[obj_name] = clustered_layers
            else:
                # 单层特征聚类
                if features.shape[0] > n_clusters:
                    if method == 'kmeans':
                        features_cpu = features.cpu().numpy()
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features_cpu)
                        clustered_features[obj_name] = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
                    elif method == 'adaptive_pooling':
                        n_features = features.shape[0]
                        if n_features > n_clusters:
                            pooled = F.adaptive_avg_pool1d(
                                features.permute(1, 0).unsqueeze(0), 
                                n_clusters
                            ).squeeze(0).permute(1, 0)
                            clustered_features[obj_name] = pooled
                        else:
                            clustered_features[obj_name] = features
                    else:
                        # 默认使用简单的随机采样
                        indices = torch.randperm(features.shape[0])[:n_clusters]
                        clustered_features[obj_name] = features[indices]
                else:
                    clustered_features[obj_name] = features
                    
        self.clustered_memory_features = clustered_features
        return clustered_features

    def get_memory_features(self, obj_name, use_clustered=False):
        """
        Retrieve memory features, optionally using clustered versions
        """
        if use_clustered and obj_name in self.clustered_memory_features:
            return self.clustered_memory_features[obj_name]
        elif obj_name in self.memory_features:
            return self.memory_features[obj_name]
        else:
            return None

    def calculate_anomaly_score_with_memory(self, patch_tokens, obj_name, use_clustered=False):
        """
        Calculate anomaly score using Memory Bank (GPU optimized version)
        """
        memory_features = self.get_memory_features(obj_name, use_clustered)
        
        if memory_features is None:
            return None
            
        # Process patch tokens, ensuring they are on the correct device
        if isinstance(patch_tokens, list):
            # Multi-layer feature case
            anomaly_maps_few_shot = []
            for idx, p in enumerate(patch_tokens):
                # Ensure on the correct device
                p = p.to(self.device)
                if isinstance(memory_features, list):
                    mem_feat = memory_features[idx].to(self.device)
                else:
                    mem_feat = memory_features.to(self.device)
                
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
        
    def combine_classification_and_segmentation(self, image_features, patch_tokens, text_features, memory_bank=None, obj_name=None, use_clustered=False):
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
            memory_score = memory_bank.calculate_anomaly_score_with_memory(patch_tokens, obj_name, use_clustered=use_clustered)
            if memory_score is not None:
                # Combine memory bank score with existing score
                memory_score_normalized = (memory_score - memory_score.min()) / (memory_score.max() - memory_score.min() + 1e-8)
                anomaly_score_combined = self.alpha * anomaly_score[:, 1] + (1 - self.alpha) * memory_score_normalized.mean()
                return anomaly_maps, anomaly_score_combined
        
        return anomaly_maps, anomaly_score[:, 1]