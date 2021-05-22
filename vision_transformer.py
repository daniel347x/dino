# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
from re import L

import torch
import torch.nn as nn

from utils import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, return_both=False):
        if return_both:
            return_attention = False
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_both:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        num_patches = num_patches_h * num_patches_w
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def _conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )


def _conv_trans_block(in_dim, out_dim, act_fn, kernel_size=3, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        ),
        act_fn,
    )


def _conv_block_2(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        _conv_block(in_dim, in_dim, act_fn, kernel_size, stride, padding),
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
    )


class _Conv_residual_conv_2(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(_Conv_residual_conv_2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.f_quant = torch.nn.quantized.FloatFunctional()
        act_fn = act_fn

        self.conv_1 = _conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = _conv_block_2(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = _conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, inputs):
        conv_1 = self.conv_1(inputs)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class _SegMap_TransConv(nn.Module):
    def __init__(self, embed_dim, start_channels, patch_size):
        super(_SegMap_TransConv, self).__init__()
        self.start_channels = start_channels
        self.bridge1 = Mlp(embed_dim, out_features=embed_dim//4, act_layer=nn.GELU, drop=0.1)
        self.bridge2 = Mlp(embed_dim//4, out_features=start_channels, act_layer=nn.GELU, drop=0.1)
        out_size = 1
        out_size *= 2
        in_channels = start_channels
        out_channels = start_channels // out_size
        self.conv1 = _conv_trans_block(in_channels, out_channels, nn.ReLU(), kernel_size=2, stride=2, padding=0) # 2x2
        if out_size < patch_size:
            in_channels = start_channels // out_size
            out_size *= 2
            out_channels = start_channels // out_size
            self.conv2 = _conv_trans_block(in_channels, out_channels, nn.ReLU(), kernel_size=2, stride=2, padding=0) # 4x4
        else:
            self.conv2 = None
        if out_size < patch_size:
            in_channels = start_channels // out_size
            out_size *= 2
            out_channels = start_channels // out_size
            self.conv3 = _conv_trans_block(in_channels, out_channels, nn.ReLU(), kernel_size=2, stride=2, padding=0) # 8x8
        else:
            self.conv3 = None
        if out_size < patch_size:
            in_channels = start_channels // out_size
            out_size *= 2
            out_channels = start_channels // out_size
            self.conv4 = _conv_trans_block(in_channels, out_channels, nn.ReLU(), kernel_size=2, stride=2, padding=0) # 16x16
        else:
            self.conv4 = None
        if out_size < patch_size:
            in_channels = start_channels // out_size
            out_size *= 2
            out_channels = start_channels // out_size
            self.conv5 = _conv_trans_block(in_channels, out_channels, nn.ReLU(), kernel_size=2, stride=2, padding=0) # 32x32
        else:
            self.conv5 = None
        in_channels = start_channels // out_size
        out_channels = 16
        self.conv_finalbridge = _conv_block_2(in_channels, out_channels, nn.ReLU(), kernel_size=1) # Drop channel count; image size should now be patch_size * patch_size
        in_channels = out_channels
        in_dim = in_channels*patch_size*patch_size
        self.final_mlp1 = Mlp(in_dim, in_dim, out_features=in_dim, act_layer=nn.GELU, drop=0.1)
        self.final_mlp2 = Mlp(in_dim, in_dim, out_features=in_dim, act_layer=nn.GELU, drop=0.1)
        self.final_conv_in_channels = 16
        out_channels = 1
        self.conv_final = _conv_block_2(self.final_conv_in_channels, out_channels, nn.ReLU(), kernel_size=1) # Drop down to 1 channel; image size should now be patch_size * patch_size
        self.patch_size = patch_size

    def forward(self, x):
        bridge = self.bridge1(x)
        out_segmaps = self.bridge2(bridge)
        patch_count = out_segmaps.size(1)
        out_segmaps = out_segmaps.reshape((out_segmaps.size(0), patch_count, self.start_channels, 1, 1))
        # bs * ncrops, patch count, 1 channel, patch size, patch size
        segmentation_patches_out = torch.zeros((out_segmaps.size(0), patch_count, 1, self.patch_size, self.patch_size)).to(x.device)
        for p in range(patch_count):
            patch_segmap = out_segmaps[:, p, :, :, :]
            if self.conv1:
                patch_segmap = self.conv1(patch_segmap)
            if self.conv2:
                patch_segmap = self.conv2(patch_segmap)
            if self.conv3:
                patch_segmap = self.conv3(patch_segmap)
            if self.conv4:
                patch_segmap = self.conv4(patch_segmap)
            if self.conv5:
                patch_segmap = self.conv(patch_segmap)
            patch_segmap = self.conv_finalbridge(patch_segmap)
            patch_segmap = self.final_mlp1(patch_segmap.reshape((patch_segmap.size(0), patch_segmap.size(1)*patch_segmap.size(2)*patch_segmap.size(3))))
            patch_segmap = self.final_mlp2(patch_segmap)
            patch_segmap = self.conv_final(patch_segmap.reshape(patch_segmap.size(0), self.final_conv_in_channels, self.patch_size, self.patch_size))
            segmentation_patches_out[:, p, :, :, :] = patch_segmap
        return segmentation_patches_out

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[256,192], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, include_segmap=False, use_segmap=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.include_segmap = include_segmap # So that teacher and student have same parameters, even if unused by teacher
        self.use_segmap = use_segmap
        if self.use_segmap:
            self.include_segmap = True

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


        # Also output patchwise segmaps
        if self.include_segmap:
            self.patch_size = patch_size
            self.img_size = img_size
            self.segmentation_class_count_including_none = 4 # one per class, hard-coded for prototype
            start_channels = 1024
            self.seg_outs = nn.ModuleList([])
            for _ in range(self.segmentation_class_count_including_none):
                self.seg_outs.append(_SegMap_TransConv(embed_dim, start_channels, patch_size))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, self.num_patches_h, self.num_patches_w, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / self.num_patches_w, h0 / self.num_patches_h),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, h, w)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        vit_cls_output_logits = x[:, 0]

        # Also output patchwise segmaps
        if self.use_segmap:
            # bs * ncrops, patch index, logits
            segmap_input = x[:, 1:, :]
            segmentations = []
            for seg_out in self.seg_outs:
                # bs * ncrops, patch count, 1 channel, patch size, patch size
                segmentation_pieces = seg_out(segmap_input)
                # bs * ncrops, 1 channel, full image height, full image width
                segmentation = torch.zeros((segmentation_pieces.size(0), 1, self.num_patches_h * self.patch_size, self.num_patches_w * self.patch_size)).to(x.device)
                # Merge pieces into a single image
                for h in range(self.num_patches_h):
                    for w in range(self.num_patches_w):
                        current_piece = h * self.num_patches_w + w
                        segmentation[:, 0, h * self.patch_size : (h+1) * self.patch_size, w * self.patch_size : (w+1) * self.patch_size] = segmentation_pieces[:, current_piece, 0, :, :]
                # each segmentation is now (batch_size * ncrops) in length,
                # because PyTorch merged the 'ncrops' list dimension and the 'batch_size' tensor dimension
                # that was given as input to forward()
                segmentations.append(segmentation)
            return vit_cls_output_logits, segmentations[0], segmentations[1], segmentations[2], segmentations[3]
        else:
            return vit_cls_output_logits


    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return_both = self.use_segmap
                blk_output = blk(x, return_attention=True, return_both=return_both)
                if return_both is False:
                    attn = blk_output
                    return attn
        x, attn = blk_output
        x = self.norm(x)

        segmap_input = x[:, 1:, :]
        segmentations = []
        for seg_out in self.seg_outs:
            segmentation_pieces = seg_out(segmap_input)
            segmentation = torch.zeros((segmentation_pieces.size(0), 1, self.num_patches_h * self.patch_size, self.num_patches_w * self.patch_size)).to(x.device)
            for h in range(self.num_patches_h):
                for w in range(self.num_patches_w):
                    current_piece = h * self.num_patches_w + w
                    segmentation[:, 0, h * self.patch_size : (h+1) * self.patch_size, w * self.patch_size : (w+1) * self.patch_size] = segmentation_pieces[:, current_piece, 0, :, :]
            segmentations.append(segmentation)

        return attn, segmentations[0], segmentations[1], segmentations[2], segmentations[3]


    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def deit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, use_segmap=False):
        super().__init__()
        self.use_segmap = use_segmap
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_segmap:
            x, segmaps_0, segmaps_1, segmaps_2, segmaps_3 = x
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        if self.use_segmap:
            # print(f'******************************')
            # print(f'In DINOHead: type(segmaps): {type(segmaps)}')
            # print(f'In DINOHead: type(segmaps[0]): {type(segmaps[0])}')
            # print(f'******************************')
            return x, segmaps_0, segmaps_1, segmaps_2, segmaps_3
        else:
            return x
