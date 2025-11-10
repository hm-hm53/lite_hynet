import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from fvcore.nn import flop_count, parameter_count
class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sg = nn.Sigmoid()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([x_avg, x_max], dim=1)
        out = self.sg(self.conv_squeeze(x))
        return out

class LKattention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # self.conv0 = nn.Conv2d(dim, dim, 1, groups=dim)
        # self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=2, groups=dim, dilation=2)
        # self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.ChannelAttention = ChannelAttention(dim)
        self.SpatialAttention = SpatialAttention()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        sig = self.SpatialAttention(attn)


        attn_spatial1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn_spatial2 = attn2 * sig[:, 1, :, :].unsqueeze(1)


        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        attn_spatial = fuse_weights[0] * attn_spatial1 + fuse_weights[1] * attn_spatial2

        attn_channel = self.ChannelAttention(attn)
        attn_spatial = self.conv(attn_spatial)

        x_out = attn_spatial * x + x * attn_channel

        return x_out

class LKattention_Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.1,
                 act_layer=nn.GELU,
                 drop_path=0.1,
                 norm_layer=nn.BatchNorm2d,
                 ):
        super().__init__()
        self.mixer = LKattention(dim)
        # 随机路劲失效
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # 构建MLp
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



if __name__ == '__main__':
    x = torch.randn(1, 64, 224, 224).cuda()
    attion =LKattention_Block(64).cuda()
    y = attion(x)
    print(y.shape)

    input = torch.randn((1, 64, 1024, 1024), device=next(attion.parameters()).device)
    params = parameter_count(attion)[""]
    Gflops, unsupported = flop_count(model=attion, inputs=(input,))
    print('GFLOPs: ', sum(Gflops.values()), 'G')
    print('Params: ', params / 1e6, 'M')

    total_frames = 1000

    start_time = time.time()
    with torch.no_grad():
        for _ in range(total_frames):
            output = attion(input)
    end_time = time.time()

    fps = total_frames / (end_time - start_time)
    print(f"FPS: {fps}")

