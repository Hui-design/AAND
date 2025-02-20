from collections import OrderedDict
from typing import Tuple, Union
import os, pdb, sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import pdb
from models.recons_net import patch_to_tensor, tensor_to_patch
from clip.model import LayerNorm, QuickGELU 


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # pdb.set_trace()
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    # def forward(self, x: torch.Tensor):
    #     return self.resblocks(x)

    def forward(self, x, f_list=[]):
        # pdb.set_trace()
        out_tokens = []
        idx = 0
        for r in self.resblocks:
            # pdb.set_trace()
            idx+=1
            x = r(x)
            if idx in f_list:
                if len(x)==2:
                    out_tokens.append(x[0])
                    out_tokens.append(x[1])
                else:
                    out_tokens.append(x.permute(1,0,2))
        return x, out_tokens #self.resblocks(x)


class de_VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (input_resolution // patch_size)  # modify
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, f_list: list):
        # x = self.conv1(x)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = tensor_to_patch(x) + self.positional_embedding.to(x.dtype)[None,1:]
        # pdb.set_trace()
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, patch_tokens = self.transformer(x, f_list)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        # # pdb.set_trace()
        # if self.proj is not None:
        #     x = x @ self.proj

        patch_tokens = [patch_to_tensor(x) for x in patch_tokens]

        return x, patch_tokens


if __name__ == '__main__':
    x = torch.randn((1,768,16,16))
    decoder_SS = de_VisionTransformer(input_resolution=256, patch_size=16, width=768, layers=6, heads=12, output_dim=512)
    # pdb.set_trace()
    _, x_new = decoder_SS(x, f_list=[2,4,6])