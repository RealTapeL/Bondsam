import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ScaledDotProductAttention(nn.Module):
    """标准缩放点积注意力"""
    def __init__(self, dropout: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, n_heads, seq_len, d_k)
            key: (batch_size, n_heads, seq_len, d_k)
            value: (batch_size, n_heads, seq_len, d_v)
            mask: (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重到值
        output = torch.matmul(attn, value)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = self.d_k
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 残差连接
        residual = query
        
        # 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展维度以适应多头
            
        # 应用注意力机制
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最后的线性变换
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class ChannelAttention(nn.Module):
    """通道注意力机制 (SE注意力的一种变体)"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CoordinateAttention(nn.Module):
    """坐标注意力机制"""
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AnomalyAttention(nn.Module):
    """专为异常检测设计的注意力机制"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 异常感知的注意力组件
        self.q_conv = nn.Conv1d(d_model, d_model, 1)
        self.k_conv = nn.Conv1d(d_model, d_model, 1)
        self.v_conv = nn.Conv1d(d_model, d_model, 1)
        
        self.attention = ScaledDotProductAttention(dropout)
        
        # 异常评分网络
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - 输入特征
        Returns:
            output: (batch_size, seq_len, d_model) - 注意力增强后的特征
            anomaly_scores: (batch_size, seq_len) - 异常评分
        """
        batch_size, seq_len, d_model = x.shape
        
        # 转换为适合卷积的形状
        x_trans = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # 生成Q, K, V
        Q = self.q_conv(x_trans).transpose(1, 2)  # (batch_size, seq_len, d_model)
        K = self.k_conv(x_trans).transpose(1, 2)  # (batch_size, seq_len, d_model)
        V = self.v_conv(x_trans).transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # 分割为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力机制
        attn_output, _ = self.attention(Q, K, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 计算异常评分
        anomaly_scores = self.anomaly_scorer(attn_output).squeeze(-1)  # (batch_size, seq_len)
        
        return attn_output, anomaly_scores


class AttentionEnhancedFeatureExtractor(nn.Module):
    """注意力增强的特征提取器"""
    def __init__(self, in_channels, out_channels, use_cbam=True, use_coord=True):
        super(AttentionEnhancedFeatureExtractor, self).__init__()
        self.use_cbam = use_cbam
        self.use_coord = use_coord
        
        # 基础卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力模块
        if use_cbam:
            self.cbam = CBAM(out_channels)
        
        if use_coord:
            self.coord_attention = CoordinateAttention(out_channels, out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_cbam:
            x = self.cbam(x)
            
        if self.use_coord:
            x = self.coord_attention(x)
            
        return x