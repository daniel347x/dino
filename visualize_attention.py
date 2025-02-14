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
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits


def apply_mask(image, mask, color, alpha=0.5, include_img=True):
    for c in range(3):
        if include_img:
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        else:
            image[:, :, c] = mask * color[c] * 255
    return image


def get_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def create_save_image_grid(imgs, save_as_png_pathname, rows=4, cols=8):
    assert rows*cols >= len(imgs)
    import matplotlib.pyplot as plt
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    if save_as_png_pathname:
        plt.imsave(save_as_png_pathname, grid)
    return grid

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5, include_img=True):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = get_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha, include_img=include_img)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    with open(fname, 'rb') as f:
        pil_img = Image.open(f)
        pil_img = pil_img.convert('RGB')
    return pil_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['deit_tiny', 'deit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--inc_segmentation', type=utils.bool_flag, default=False, help="""Whether or not model was trained with SSL segmentation""")
    parser.add_argument('--device', default='cuda', type=str, help="""CUDA or CPU""")
    args = parser.parse_args()

    device = torch.device(args.device)
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, use_segmap=args.inc_segmentation)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "deit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "deit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    h, w = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :h, :w].unsqueeze(0)

    h_featmap = img.shape[-2] // args.patch_size
    w_featmap = img.shape[-1] // args.patch_size
    n_patches = h_featmap * w_featmap

    if args.inc_segmentation:
        attentions, segmaps_0, segmaps_1, segmaps_2, segmaps_3 = model.get_last_selfattention(img.to(device))
        # model.out_dim == 4: hard-coded currently to "seg_classes": ["None", "Chart", "List", "Table"] from the config settings
        assert segmaps_0.shape == (1, 1, h, w) # etc
    else:
        attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # attentions.shape: batch_size=1, number_heads, patch_count as OUTPUT (includes CLS token), patch_count as ATTENTION (includes CLS token)
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, h_featmap, w_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, h_featmap, w_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        # print(f"{fname} saved.")

    image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))

    pil_imgs = []
    tmp_filename = os.path.join(args.output_dir, 'tmp.png')
    for j in range(nh):
        pil_img = display_instances(image, th_attn[j], fname=tmp_filename, blur=False)
        pil_imgs.append(pil_img)
    grid = create_save_image_grid(pil_imgs, os.path.join(args.output_dir, f"img_grid_attentions_th{args.threshold}.png"), rows=3, cols=4)
    if args.inc_segmentation:
        pil_imgs_segmaps = []
        segmap_idx = 0
        pil_img = display_instances(image, segmaps_0[0].squeeze(0).numpy(), fname=tmp_filename, blur=False, include_img=False)
        pil_imgs_segmaps.append(pil_img)
        segmap_idx = 1
        pil_img = display_instances(image, segmaps_1[0].squeeze(0).numpy(), fname=tmp_filename, blur=False, include_img=False)
        pil_imgs_segmaps.append(pil_img)
        segmap_idx = 2
        pil_img = display_instances(image, segmaps_2[0].squeeze(0).numpy(), fname=tmp_filename, blur=False, include_img=False)
        pil_imgs_segmaps.append(pil_img)
        segmap_idx = 3
        pil_img = display_instances(image, segmaps_3[0].squeeze(0).numpy(), fname=tmp_filename, blur=False, include_img=False)
        pil_imgs_segmaps.append(pil_img)
        grid = create_save_image_grid(pil_imgs_segmaps, os.path.join(args.output_dir, f"img_grid_segmaps.png"), rows=2, cols=2)
