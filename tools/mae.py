import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class MAEPretrainer:
    """
    掩码自编码器预训练器
    """
    def __init__(self, model, masking_ratio=0.75, device='cuda'):
        self.model = model
        self.masking_ratio = masking_ratio
        self.device = device
        
    def patchify(self, imgs):
        """
        将图像分割成patches
        """
        p = 16  # patch size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        将patches重新组合成图像
        """
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        随机遮蔽patches
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def mae_forward(self, imgs):
        """
        MAE前向传播
        """
        # 获取图像patches
        x = self.patchify(imgs)
        # 随机遮蔽
        x_masked, mask, ids_restore = self.random_masking(x, self.masking_ratio)
        
        # 使用模型编码器处理图像以获取特征
        with torch.no_grad():
            # 提取特征，这里我们只使用图像特征
            image_features, patch_tokens, _ = self.model.clip_model.extract_feat(
                imgs, ["dummy"], fastsam_mask=None
            )
        
        # 处理patch_tokens以获取合适的维度
        if isinstance(patch_tokens, list):
            # 如果是多层特征，使用最后一层
            token_features = patch_tokens[-1]
        else:
            token_features = patch_tokens
            
        # 确保token_features是正确的形状 [B, N, D]
        if token_features.dim() == 3:
            # 已经是正确的形状
            latent = token_features
        elif token_features.dim() == 4:
            # 如果是 [B, C, H, W] 形状，需要展平
            B, C, H, W = token_features.shape
            latent = token_features.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        else:
            # 其他情况，使用image_features
            if image_features.dim() == 2:
                # [B, D] -> [B, 1, D]
                latent = image_features.unsqueeze(1)
            else:
                latent = image_features
        
        # 简化的解码过程
        # 为了简化，我们直接使用latent的均值来预测所有patch
        latent_mean = latent.mean(dim=1)  # [B, D]
        
        # 解码器部分 - 映射到patch维度
        decoder = nn.Linear(latent_mean.shape[-1], x.shape[-1]).to(latent_mean.device)
        # 扩展到所有patch
        pred = decoder(latent_mean).unsqueeze(1).repeat(1, x.shape[1], 1)  # [B, L, D]
        
        return pred, mask

    def mae_loss(self, imgs, pred, mask):
        """
        MAE损失函数
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def pretrain_mae(self, dataloader, epochs=10, lr=1e-4, logger=None):
        """
        执行MAE预训练
        """
        # 创建解码器
        decoder = nn.Linear(768, 768).to(self.device)  # 适配ViT-B-16的维度
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(decoder.parameters()), 
                                     lr=lr, betas=(0.9, 0.95))
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 创建进度条
            pbar = tqdm(dataloader, desc=f'MAE Pretraining Epoch {epoch+1}/{epochs}')
            
            for batch_idx, items in enumerate(pbar):
                imgs = items['img'].to(self.device)
                
                try:
                    # MAE前向传播
                    pred, mask = self.mae_forward(imgs)
                    
                    # 计算损失
                    loss = self.mae_loss(imgs, pred, mask)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    if logger:
                        logger.info(f"Error in MAE pretraining batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if logger:
                logger.info(f"MAE Pretraining Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            print(f"MAE Pretraining Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


class MAEDecoder(nn.Module):
    """
    MAE解码器模块
    """
    def __init__(self, embed_dim=768, decoder_embed_dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 简化的解码器transformer层
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=8,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(4)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)
        
    def forward(self, x, ids_restore):
        # 嵌入到解码器维度
        x = self.decoder_embed(x)
        
        # 添加掩码tokens
        mask_tokens = torch.zeros(x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2], 
                                 device=x.device)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # 应用解码器transformer块
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 预测像素值
        x = self.decoder_pred(x)
        
        return x


def setup_mae_pretrainer(model, masking_ratio=0.75):
    """
    创建MAE预训练器的便捷函数
    """
    device = next(model.parameters()).device
    return MAEPretrainer(model, masking_ratio, device)


def perform_mae_pretraining(model, train_data, batch_size=8, epochs=10, lr=1e-4, 
                           masking_ratio=0.75, logger=None):
    """
    执行MAE预训练的便捷函数
    """
    # 创建数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建MAE预训练器
    mae_pretrainer = setup_mae_pretrainer(model, masking_ratio)
    
    # 执行预训练
    mae_pretrainer.pretrain_mae(train_dataloader, epochs, lr, logger)
    
    return mae_pretrainer