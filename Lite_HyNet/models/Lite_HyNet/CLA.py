import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
import copy


class DynamicAxialPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_shape: int = 512):
        super().__init__()
        self.dim = dim
        self.max_shape = max_shape
        self.pos_embed = nn.Parameter(torch.randn([1, dim, max_shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        if N != self.max_shape:
            x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        else:
            x = x + self.pos_embed
        return x


class AxialPositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_weight = nn.Parameter(torch.ones(1, 128, 1, 1))  # 行位置权重
        self.col_weight = nn.Parameter(torch.ones(1, 128, 1, 1))  # 列位置权重

    def forward(self, x_row, x_column, v):

        row_map = x_row.expand(-1, -1, v.shape[-2], -1)
        col_map = x_column.expand(-1, -1, -1, v.shape[-1])

        return v+self.row_weight * row_map + self.col_weight * col_map


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

class MCFA(nn.Module):
    def __init__(self,
                 hidden_dim,
                 key_dim,
                 current_d_dim,
                 sr_ratio=1,
                 num_heads=8,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.current_d_dim = current_d_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d_norm = LayerNorm2d(current_d_dim)
        self.d_proj = nn.Sequential(
            nn.Conv2d(current_d_dim, current_d_dim, kernel_size=3, padding=1, groups=current_d_dim,
                      bias=False),
            nn.BatchNorm2d(current_d_dim),
            nn.GELU(),
            nn.Conv2d(current_d_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
        )

        self.q = nn.Sequential(
            self.get_sr(hidden_dim, nh_kd, sr_ratio)
        )
        self.k = nn.Sequential(
            self.get_sr(hidden_dim, nh_kd, sr_ratio)
        )
        self.v = nn.Sequential(
            nn.Conv2d(hidden_dim, nh_kd, kernel_size=1, bias=False),
            nn.BatchNorm2d(nh_kd),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim+2*nh_kd, hidden_dim+2*nh_kd, kernel_size=3, padding=1, groups=hidden_dim+2*nh_kd),
            nn.GELU(),
            nn.Conv2d(hidden_dim+2*nh_kd, hidden_dim, kernel_size=1),
        )

        self.pos_emb_rowq = DynamicAxialPositionalEmbedding(nh_kd)
        self.pos_emb_rowk = DynamicAxialPositionalEmbedding(nh_kd)
        self.pos_emb_columnq = DynamicAxialPositionalEmbedding(nh_kd)
        self.pos_emb_columnk = DynamicAxialPositionalEmbedding(nh_kd)
        self.act = activation()
        self.AxialPositionEncoding = AxialPositionEncoding()

    def forward(self, x, shortcut):
        B, C, H, W = x.shape
        s = self.d_proj(self.d_norm(shortcut))
        k, v = torch.chunk(s, 2, dim=1)


        q = self.q(x)
        k = self.k(k)
        v = self.v(v)

        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, self.key_dim, H)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, self.key_dim, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, self.key_dim, H)

        attn_row = F.scaled_dot_product_attention(qrow, krow, vrow).reshape(B, self.nh_kd, 1, H)


        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, self.key_dim, W)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, self.key_dim, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, self.key_dim, W)


        attn_column = F.scaled_dot_product_attention(qcolumn, kcolumn, vcolumn).reshape(B, self.nh_kd, 1, W)


        attn = self.AxialPositionEncoding(attn_row, attn_column, v)
        x = torch.cat([x, attn, v.reshape(B, -1, H, W)], dim=1)
        x = rearrange(x, 'b (c g) h w -> b (g c) h w', g=5).contiguous()  ### channel shuffle for efficiency
        x = self.proj(x)

        return self.act(x)
    @staticmethod
    def get_sr(a, b, sr_ratio):
        sr = nn.Sequential(
                nn.Conv2d(a, b, kernel_size=3, stride=1, dilation=sr_ratio, padding=sr_ratio, groups=a,
                          bias=False),
                nn.BatchNorm2d(b),
                nn.GELU(),
            )
        return sr


if __name__ == '__main__':
    x = torch.randn(2, 64, 224, 224)
    shortcut = torch.randn(2, 130, 224, 224)
    net = MCFA(64, 16, 130)
    y = net(x, shortcut)
    print(y.shape)
