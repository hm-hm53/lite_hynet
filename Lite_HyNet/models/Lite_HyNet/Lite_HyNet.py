import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Union, Type, List, Tuple, Callable, Dict, Any
from u_kan import KANBlock
from fvcore.nn import flop_count, parameter_count
import copy
from LKattention import LKattention_Block, Mlp
from vmamba import VSSM, VSSBlock, Permute, SS2D
from CLA import MCFA
from torch.utils import checkpoint
from models.datasets.vaihingen_dataset import *
from torch.utils.data import DataLoader

from tqdm import tqdm
import time

def regnety(pretrained=False, **kwargs):
    model = timm.create_model("regnety_016.tv2_in1k", features_only=True, output_stride=32,
                                      out_indices=(1, 2, 3, 4),
                                      pretrained=False)  # , pretrained_cfg_overlay=dict(file='~/.cache/huggingface/hub/model–timm–resnet18.a1_in1k/pytorch_model.bin'))        # build decoder layers
    state_dict = torch.load('/home/yons/文档/Euntmamba_new/pytorch_model.bin')
    msg = model.load_state_dict(state_dict, strict=False)
    print('[INFO] ', msg)
    return model

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Res_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = Conv(in_channels, out_channels, kernel_size=1)
        self.proj = ConvBNReLU(out_channels+2, out_channels+2, kernel_size=3)
    def forward(self, x):
        pad1 = torch.mean(x, dim=1, keepdim=True)
        pad2, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv0(x)
        x = torch.cat((x, pad1, pad2), dim=1)
        x = rearrange(x, 'b (c g) h w -> b (g c) h w', g=3).contiguous()  ### channel shuffle for efficiency
        x = self.proj(x)
        return x


class MFKAN(nn.Module):
    def __init__(self,
                 encoder_dims=(48, 120, 336, 888),
                 decoder_dims=64,
                 embed_dim=66,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=(1, 1, 1),
                 **kwargs
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_1x1 = nn.Conv2d(encoder_dims[3], decoder_dims, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(embed_dim//3, embed_dim//3, kernel_size=3, stride=1, padding=1, groups=embed_dim//3)
        self.conv_5x5 = nn.Conv2d(embed_dim//3, embed_dim//3, kernel_size=5, stride=1, padding=2, groups=embed_dim//3)
        self.conv_7x7 = nn.Conv2d(embed_dim//3, embed_dim//3, kernel_size=7, stride=1, padding=3, groups=embed_dim//3)
        self.pos_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, decoder_dims, kernel_size=3, stride=1, padding=1)
        self.gelu = nn.GELU()
        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.Tok_KANblock = nn.ModuleList([KANBlock(
            dim=embed_dim,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

    def forward(self, x):
        B = x.shape[0]
        pad1 = torch.mean(x, dim=1, keepdim=True)
        pad2, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv_1x1(x)
        x_f = torch.cat((x, pad1, pad2), dim=1)
        x_f = rearrange(x_f, 'b (c g) h w -> b (g c) h w', g=3).contiguous()  ### channel shuffle for efficiency
        x1 = self.conv_3x3(x_f[:, :self.embed_dim//3, :, :])
        x2 = self.conv_5x5(x_f[:, self.embed_dim//3:2*self.embed_dim//3, :, :])
        x3 = self.conv_7x7(x_f[:, 2*self.embed_dim//3:, :, :])
        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = x_fused + self.pos_embed(x_fused)
        _, _, H, W = x_fused.shape
        out = x_fused.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.Tok_KANblock):
            out = blk(out, H, W)
        out = self.norm(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output = self.gelu(self.conv_last(out)) + x
        shortcut = output
        return (output, shortcut)

class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            attn_drop=0.1,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpwoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x.permute(0, 3, 1, 2).contiguous()

class VSSblock(nn.Module):
    def __init__(self,
                 dim):
        super().__init__()
        self.VSSlayer = VSSLayer(dim)
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        input, shortcut = x
        input = input+self.pos_embed(input)
        input = self.VSSlayer(input)
        return (input, shortcut)


class MF_VSS(nn.Module):
    def __init__(
            self,
            hidden_dim,
            current_d_dim,
            drop_path: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0.1,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # ==========================
            key_dim=16,
            sr_ratio=1,
            dropout=0.1,
            **kwargs
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.act = nn.GELU
        self.dropout = nn.Dropout(dropout)
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        self.drop_path = DropPath(drop_path)
        self.mcfa = MCFA(hidden_dim, key_dim, current_d_dim, sr_ratio)

    def forward(self, input, shortcut):
        x = self.mcfa(input, shortcut)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.op(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)

        return x


class MFVSS_Block(nn.Module):
    def __init__(
            self,
            hidden_dim,
            current_d_dim,
            res_in_channels,
            use_checkpoint=False,
            drop_path=0.1,
            mlp_ratio=4.,
            norm_layer=LayerNorm2d,
            sr_ratio=1,
            ls_init_value=1e-5,
            is_last: bool = False,
            is_first: bool = False,
            **kwargs,
    ):

        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm = norm_layer(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)
        self.MF_VSS = MF_VSS(hidden_dim, current_d_dim, sr_ratio=sr_ratio)
        self.lk_attention = LKattention_Block(hidden_dim)
        self.pos_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, drop=drop_path)
        self.Res_conv = Res_conv(res_in_channels, hidden_dim)
        self.is_last = is_last
        self.is_first = is_first
        self.use_checkpoint = use_checkpoint

        if ls_init_value is not None:
            self.layerscale_1 = LayerScale(hidden_dim, init_value=ls_init_value)
            self.layerscale_2 = LayerScale(hidden_dim, init_value=ls_init_value)
        else:
            self.layerscale_1 = nn.Identity()
            self.layerscale_2 = nn.Identity()


    def _forward(self, x, shortcut):
        x = x + self.pos_embed(x)
        if self.is_last:
            x = self.lk_attention(x)
            shortcut = x
        else:
            x = self.layerscale_1(x) + self.drop_path(self.MF_VSS(self.norm(x), shortcut))  # Token Mixer
            x = self.layerscale_2(x) + self.drop_path(self.mlp(self.norm2(x)))  # FFN
            shortcut = x

        return (x, shortcut)

    def forward(self, x, res, shortcut_mfkm):
        res = self.Res_conv(res)
        _, _, h, w = res.shape
        shortcut_mfkm = F.interpolate(shortcut_mfkm, size=(h, w), mode='bilinear', align_corners=False)
        if self.is_first:
            input = x
            shortcut = torch.cat((res, shortcut_mfkm), dim=1)
            input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            input, shortcut = x
            input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
            shortcut = F.interpolate(shortcut, scale_factor=2, mode='bilinear', align_corners=False)
            shortcut = torch.cat((shortcut, res, shortcut_mfkm), dim=1)

        if self.use_checkpoint and input.requires_grad:
            return checkpoint.checkpoint(self._forward, input, shortcut, use_reentrant=True)
        else:
            return self._forward(input, shortcut)

class Auxhead(nn.Module):
        def __init__(self, in_channels=64, num_classes=6):
            super().__init__()
            self.conv0 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            self.conv_spatial = nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=6, groups=in_channels, dilation=3, bias=False)
            self.drop = nn.Dropout(0.1)
            self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, dilation=1, stride=1, padding=0, bias=False)

        def forward(self, x, h, w):
            feat = self.conv_spatial(self.conv0(x))
            feat = self.drop(feat)
            feat = self.conv_out(feat)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            return feat


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder_channels: Union[Tuple[int, ...], List[int]] = None,
                 current_d_dim: Union[Tuple[int, ...], List[int]] = None,
                 sr_ratio: Union[Tuple[int, ...], List[int]] = None,
                 decoder_dim=64,
                 **kwargs):
        super().__init__()
        encoder_output_channels = encoder_channels
        current_d_dim_output_channels = current_d_dim
        sr_ratio = sr_ratio
        self.num_classes = num_classes

        # stage3
        self.mfkm = MFKAN()
        self.MF_VSS1 = MFVSS_Block(
                        decoder_dim,
                        current_d_dim_output_channels[-1],
                        encoder_output_channels[-2],
                        sr_ratio=sr_ratio[-1],
                        is_first=True)
        # stage2
        self.VSS2 = VSSblock(decoder_dim)
        self.MF_VSS2 = MFVSS_Block(
                        decoder_dim,
                        current_d_dim_output_channels[-2],
                        encoder_output_channels[-3],
                        sr_ratio=sr_ratio[-2])

        # stage1
        self.VSS3 = VSSblock(decoder_dim)
        self.MF_VSS3 = MFVSS_Block(
                        decoder_dim,
                        current_d_dim_output_channels[-3],
                        encoder_output_channels[-4],
                        sr_ratio=sr_ratio[-3],
                        is_last=True)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = Auxhead(decoder_dim, num_classes)

        self.segmentation_head = nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
                                               nn.BatchNorm2d(decoder_dim),
                                               nn.GELU(),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               nn.Conv2d(decoder_dim, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, res1, res2, res3, res4, h, w):

        if self.training:

            x, shortcut_mfkm = self.mfkm(res4)
            h4 = self.up4(x)
            x = self.MF_VSS1(x, res3, shortcut_mfkm)

            x = self.VSS2(x)
            up3, _ = x
            h3 = self.up3(up3)
            x = self.MF_VSS2(x, res2, shortcut_mfkm)

            x = self.VSS3(x)
            up2, _ = x
            h2 = up2
            x, _ = self.MF_VSS3(x, res1, shortcut_mfkm)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x, shortcut_mfkm = self.mfkm(res4)
            x = self.MF_VSS1(x, res3, shortcut_mfkm)

            x = self.VSS2(x)
            x = self.MF_VSS2(x, res2, shortcut_mfkm)

            x = self.VSS3(x)
            x, _ = self.MF_VSS3(x, res1, shortcut_mfkm)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x



class Lite_HyNet(nn.Module):
    def __init__(self,
                 num_classes=6,
                 **kwargs
                 ):
            super().__init__()
            self.encoder = regnety()
            encoder_channels = self.encoder.feature_info.channels()
            current_d_dim = [194, 194, 130]
            sr_ratio = [3, 2, 1]
            # encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
            self.decoder = Decoder(num_classes=num_classes,
                                   encoder_channels=encoder_channels,
                                   current_d_dim=current_d_dim,
                                   sr_ratio=sr_ratio)

    def forward(self, x):
            h, w = x.size()[-2:]
            res1, res2, res3, res4 = self.encoder(x)
            if self.training:
                x, ah = self.decoder(res1, res2, res3, res4, h, w)
                return x, ah
            else:
                x = self.decoder(res1, res2, res3, res4, h, w)
                return x



if __name__ == '__main__':
    model = Lite_HyNet()
    model.cuda()
    x = torch.randn((2, 3, 1024, 1024)).cuda()
    model = copy.deepcopy(model)
    model.cuda().eval()
    input = torch.randn((1, 3, 1024, 1024), device=next(model.parameters()).device)
    params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input,))
    print('GFLOPs: ', sum(Gflops.values()), 'G')
    print('Params: ', params / 1e6, 'M')
    y, ta = model(x)
    print(y.shape)

















