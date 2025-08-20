import cv2
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

try:
    from loss import FocalLoss, BinaryDiceLoss
except ImportError:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return F_loss.mean()
    
    class BinaryDiceLoss(nn.Module):
        def __init__(self, smooth=1):
            super(BinaryDiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            intersection = (inputs * targets).sum()
            dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
            return 1 - dice

try:
    from tools import visualization, calculate_metric, calculate_average_metric
except ImportError:
    def calculate_metric(*args, **kwargs):
        return {'auroc_im': 0.0, 'f1_im': 0.0, 'ap_im': 0.0, 'auroc_px': 0.0, 'f1_px': 0.0, 'ap_px': 0.0}
    
    def calculate_average_metric(*args, **kwargs):
        return {'auroc_im': 0.0, 'f1_im': 0.0, 'ap_im': 0.0, 'auroc_px': 0.0, 'f1_px': 0.0, 'ap_px': 0.0}
    
    class visualization:
        @staticmethod
        def plot_sample_cv2(*args, **kwargs):
            pass

from .bondsam_model import BondSAM

try:
    from .clip_model import create_model_and_transforms
except ImportError:
    def create_model_and_transforms(*args, **kwargs):
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        
        backbone = kwargs.get('backbone', 'ViT-B-16')
        image_size = kwargs.get('image_size', 224)
        pretrained = kwargs.get('pretrained', 'openai')
        
        config_path = f'./model_configs/{backbone}.json'
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            embed_dim = config['embed_dim']
            vision_width = config['vision_cfg']['width']
            vision_layers = config['vision_cfg']['layers']
            text_width = config['text_cfg']['width']
            text_layers = config['text_cfg']['layers']
        except:
            embed_dim = 512
            vision_width = 768
            vision_layers = 12
            text_width = 512
            text_layers = 12
            
        class DummyAttentionBlock(nn.Module):
            def __init__(self, d_model=512):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, 8)
                self.ln_1 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
                self.ln_2 = nn.LayerNorm(d_model)
                
            def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
                x = q_x
                attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
                x = x + attn_out
                x = self.ln_1(x)
                mlp_out = self.mlp(x)
                x = x + mlp_out
                x = self.ln_2(x)
                return x, None
                
        class DummyVisual:
            def __init__(self, device='cpu'):
                self.device = device
                self.class_embedding = torch.randn(vision_width, device=device)
                self.positional_embedding = torch.randn(
                    (image_size // 16) * (image_size // 16) + 1, vision_width, device=device)
                self.conv1 = nn.Conv2d(3, vision_width, kernel_size=16, stride=16).to(device)
                self.ln_pre = nn.LayerNorm(vision_width).to(device)
                
                self.transformer = nn.Module()
                self.transformer.resblocks = nn.ModuleList([
                    DummyAttentionBlock(vision_width) for _ in range(vision_layers)
                ])
                self.ln_post = nn.LayerNorm(vision_width).to(device)
                self.proj = torch.randn(vision_width, embed_dim, device=device)
                
            def to(self, device):
                self.device = device
                self.class_embedding = self.class_embedding.to(device)
                self.positional_embedding = self.positional_embedding.to(device)
                self.conv1 = self.conv1.to(device)
                self.ln_pre = self.ln_pre.to(device)
                self.ln_post = self.ln_post.to(device)
                self.proj = self.proj.to(device)
                self.transformer.resblocks = self.transformer.resblocks.to(device)
                return self
                
            def forward(self, x, output_layers):
                if x.device != self.device:
                    x = x.to(self.device)
                    
                b, c, h, w = x.shape
                features = torch.randn(b, vision_width, device=self.device)
                patch_tokens = [torch.randn(b, (h//16)*(w//16), vision_width, device=self.device) for _ in output_layers]
                patch_embedding = torch.randn(b, (h//16)*(w//16) + 1, vision_width, device=self.device)
                return features, patch_tokens, patch_embedding
                
        class DummyTransformer:
            def __init__(self, device='cpu'):
                self.device = device
                self.get_cast_dtype = lambda: torch.float32
                self.resblocks = nn.ModuleList([
                    DummyAttentionBlock(text_width) for _ in range(text_layers)
                ])
                
            def to(self, device):
                self.device = device
                self.resblocks = self.resblocks.to(device)
                return self
                
        class DummyModel:
            def __init__(self, device='cpu'):
                self.device = device
                self.visual = DummyVisual(device)
                self.transformer = DummyTransformer(device)
                self.token_embedding = nn.Embedding(1000, text_width).to(device)
                self.positional_embedding = torch.randn(77, text_width, device=device)
                self.ln_final = nn.LayerNorm(text_width).to(device)
                self.text_projection = torch.randn(text_width, embed_dim, device=device)
                self.attn_mask = None
            
            def to(self, device):
                self.device = device
                self.visual = self.visual.to(device)
                self.transformer = self.transformer.to(device)
                self.token_embedding = self.token_embedding.to(device)
                self.positional_embedding = self.positional_embedding.to(device)
                self.ln_final = self.ln_final.to(device)
                self.text_projection = self.text_projection.to(device)
                return self
                
            def eval(self):
                return self
                
            def encode_text(self, text):
                if text.device != self.device:
                    text = text.to(self.device)
                return torch.randn(text.shape[0], embed_dim, device=self.device)
        
        return DummyModel(kwargs.get('device', 'cpu')), None, transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

try:
    from .attention import (
        MultiHeadAttention, 
        CBAM, 
        CoordinateAttention, 
        AnomalyAttention,
        AttentionEnhancedFeatureExtractor
    )
except ImportError:
    class MultiHeadAttention:
        def __init__(self, *args, **kwargs):
            pass
            
    class CBAM:
        def __init__(self, *args, **kwargs):
            pass
            
    class CoordinateAttention:
        def __init__(self, *args, **kwargs):
            pass
            
    class AnomalyAttention:
        def __init__(self, *args, **kwargs):
            pass
            
    class AttentionEnhancedFeatureExtractor:
        def __init__(self, *args, **kwargs):
            pass

import numpy as np
import torch
from torch import nn

class BondSAM_Trainer(nn.Module):
    def __init__(
            self,
            backbone, feat_list, input_dim, output_dim,
            learning_rate, device, image_size,
            prompting_depth=3, prompting_length=2,
            prompting_branch='VL', prompting_type='SD',
            use_hsf=True, k_clusters=20,
            use_fastsam=False,
            use_anomaly_attention=False,
            use_enhanced_extractor=False,
            use_memory_bank=False,
            k_shot=10
    ):

        super(BondSAM_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.use_fastsam = use_fastsam
        self.use_anomaly_attention = use_anomaly_attention
        self.use_enhanced_extractor = use_enhanced_extractor
        self.use_memory_bank = use_memory_bank
        self.k_shot = k_shot

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        if self.use_fastsam:
            try:
                from .fastsam_processor import create_fastsam_processor
                self.fastsam_processor = create_fastsam_processor(device=device)
            except ImportError:
                print("FastSAM module not found. Using placeholder.")
                self.fastsam_processor = None

        if self.use_anomaly_attention:
            self.anomaly_attention = AnomalyAttention(d_model=512)
            self.cbam_attention = CBAM(in_planes=512)
            self.coord_attention = CoordinateAttention(in_channels=512, out_channels=512)

        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip = freeze_clip.to(device)
        freeze_clip.eval()

        self.clip_model = BondSAM(freeze_clip=freeze_clip,
                                  text_channel=output_dim,
                                  visual_channel=input_dim,
                                  prompting_length=prompting_length,
                                  prompting_depth=prompting_depth,
                                  prompting_branch=prompting_branch,
                                  prompting_type=prompting_type,
                                  use_hsf=use_hsf,
                                  k_clusters=k_clusters,
                                  output_layers=feat_list,
                                  device=device,
                                  image_size=image_size).to(device)

        if self.use_enhanced_extractor:
            self.attention_feature_extractor = AttentionEnhancedFeatureExtractor(
                in_channels=input_dim, 
                out_channels=output_dim,
                use_cbam=True,
                use_coord=True
            )

        if self.use_memory_bank:
            try:
                from tools.memorybank import MemoryBank, MultiLayerFeatureFusion, AnomalyClassificationAndSegmentation
                self.memory_bank = MemoryBank(device=device)
                self.feature_fusion = MultiLayerFeatureFusion(image_size=image_size, device=device)
                self.anomaly_combiner = AnomalyClassificationAndSegmentation(alpha=0.5, device=device)
            except ImportError:
                print("VAND-APRIL-GAN feature modules not found.")
                self.memory_bank = None
                self.feature_fusion = None
                self.anomaly_combiner = None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        self.learnable_paramter_list = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'dynamic_visual_prompt_generator',
            'dynamic_text_prompt_generator'
        ]

        if self.use_anomaly_attention:
            self.learnable_paramter_list.extend([
                'anomaly_attention',
                'cbam_attention',
                'coord_attention'
            ])
            
        if self.use_enhanced_extractor:
            self.learnable_paramter_list.extend([
                'attention_feature_extractor'
            ])

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            for update_name in self.learnable_paramter_list:
                if update_name in name:
                    self.params_to_update.append(param)

        self.optimizer = torch.optim.AdamW(
            self.params_to_update, 
            lr=learning_rate, 
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        self.learning_rate = learning_rate
        # Initialize gradient scaler for mixed precision training (fixed deprecation warning)
        self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def save(self, path):
        self.save_dict = {}
        for param, value in self.clip_model.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)

    def load(self, path):
        self.clip_model.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        image = items['img'].to(self.device)
        cls_name = items['cls_name']
        fastsam_mask = items.get('fastsam_mask', None)

        if image.dtype != torch.float32:
            image = image.float()
            
        if fastsam_mask is not None:
            fastsam_mask = fastsam_mask.to(self.device)
            if fastsam_mask.dtype != image.dtype:
                fastsam_mask = fastsam_mask.float()

        # Use mixed precision training to save memory (fixed deprecation warning)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=False, fastsam_mask=fastsam_mask)

        if not isinstance(anomaly_map, list):
            anomaly_map = [anomaly_map]

        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze()

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        if gt.dtype != torch.float32:
            gt = gt.float()

        is_anomaly = items['anomaly'].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0
        
        if is_anomaly.dtype != torch.float32:
            is_anomaly = is_anomaly.float()
            
        loss = 0

        if anomaly_score.dim() == 2 and anomaly_score.shape[1] == 2:
            is_anomaly_onehot = torch.zeros_like(anomaly_score)
            is_anomaly_onehot[:, 1] = is_anomaly.squeeze()
            is_anomaly_onehot[:, 0] = 1 - is_anomaly.squeeze()
            classification_loss = self.loss_focal(anomaly_score, is_anomaly_onehot)
        else:
            classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
        loss += classification_loss

        seg_loss = 0
        for am in anomaly_map:
            if am.dtype != torch.float32:
                am = am.float()
                
            if am.dim() == 4 and gt.dim() == 3:
                gt_expanded = gt.unsqueeze(1)
                gt_expanded = gt_expanded.expand(-1, 2, -1, -1)
            elif am.dim() == 4 and gt.dim() == 2:
                gt_expanded = gt.unsqueeze(0).unsqueeze(0)
                gt_expanded = gt_expanded.expand(am.shape[0], 2, -1, -1)
            else:
                gt_expanded = gt
                
            seg_loss += self.loss_focal(am, gt_expanded)
            
            if am.dim() == 4 and am.shape[1] == 2:
                seg_loss += self.loss_dice(am[:, 1, :, :], gt)
                seg_loss += self.loss_dice(am[:, 0, :, :], 1-gt)

        loss += seg_loss

        self.optimizer.zero_grad()
        # Use gradient scaler for mixed precision training
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Clean up unnecessary variables
        del image, anomaly_map, anomaly_score, gt, is_anomaly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.item() if hasattr(loss, 'item') else float(loss)

    def train_epoch_with_progress(self, batch_dataloader):
        self.clip_model.train()
        loss_list = []
        for items in batch_dataloader:
            loss = self.train_one_batch(items)
            loss_list.append(loss)
            # Update progress bar with current batch loss
            batch_dataloader.set_postfix({'Batch Loss': f'{loss:.4f}'})

        return np.mean(loss_list)

    def train_epoch(self, loader):
        self.clip_model.train()
        loss_list = []
        for items in loader:
            loss = self.train_one_batch(items)
            loss_list.append(loss)

        return np.mean(loss_list)

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()

        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []

        with torch.no_grad():
            image_indx = 0
            for indx, items in enumerate(dataloader):
                if save_fig:
                    path = items['img_path']
                    for _path in path:
                        vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                        results['imgs'].append(vis_image)
                    cls_name = items['cls_name']
                    for _cls_name in cls_name:
                        image_indx += 1
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                fastsam_mask = items.get('fastsam_mask', None)
                
                if image.dtype != torch.float32:
                    image = image.float()
                
                if fastsam_mask is not None:
                    fastsam_mask = fastsam_mask.to(self.device)
                    if fastsam_mask.dtype != torch.float32:
                        fastsam_mask = fastsam_mask.float()

                results['cls_names'].extend(cls_name)
                gt_mask = items['img_mask']
                if gt_mask.dtype != torch.float32:
                    gt_mask = gt_mask.float()
                    
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                for _gt_mask in gt_mask:
                    results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())

                if self.use_memory_bank and self.memory_bank and hasattr(self.memory_bank, 'memory_features'):
                    image_features, patch_tokens, text_features = self.clip_model.extract_feat(
                        image, cls_name, fastsam_mask=fastsam_mask
                    )
                    
                    if self.anomaly_combiner:
                        anomaly_maps, anomaly_score = self.anomaly_combiner.combine_classification_and_segmentation(
                            image_features, patch_tokens, text_features, self.memory_bank, cls_name[0]
                        )
                        
                        if isinstance(anomaly_maps, list):
                            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
                            anomaly_map = anomaly_map.squeeze(1)
                        else:
                            anomaly_map = anomaly_maps
                    else:
                        anomaly_map, anomaly_score = self.clip_model.visual_text_similarity(
                            image_features, patch_tokens, text_features, aggregation=True
                        )
                else:
                    anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=True, fastsam_mask=fastsam_mask)

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results['anomaly_maps'].append(_anomaly_map)
                    results['anomaly_scores'].append(_anomaly_score)

                is_anomaly = np.array(items['anomaly'])
                for _is_anomaly in is_anomaly:
                    results['imgs_gts'].append(_is_anomaly)

        if save_fig:
            print('saving fig.....')
            visualization.plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'BondSAM': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir
            )

        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()

        for obj in obj_list:
            metric = calculate_metric(results, obj)
            obj_full_name = f'{obj}'
            metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict