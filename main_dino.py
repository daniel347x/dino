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
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.data.sampler import RandomSampler

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
import torch.autograd.profiler as profiler

from deepink.segmentation.config import MCv16_nma_b as config
from deepink.segmentation.data_loaders import PageLoader
from deepink.core.utils import load_pickle

DOCS = 'training'
testing_dataset_path = config['test_files']
training_dataset_path = config['train_files']
anchors = config['anchors_file']
assert DOCS in ['testing', 'training']

load_path = testing_dataset_path if DOCS == 'testing' else training_dataset_path
print(f'Loading {load_path}')
docs = load_pickle(load_path)
class_weights = torch.tensor(docs["class_weights"]).float()
docs = docs['docs']
anchors = load_pickle(anchors)
anchors = anchors["anchors"]

fscale = 1
data_loader_args = {}






torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (deit_tiny, deit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--device', default="cuda", type=str, help='CUDA or CPU?')
    parser.add_argument('--inc_segmentation', type=utils.bool_flag, default=False, help="""Whether or not to incorporate an SSL segmentation loss on the student""")
    parser.add_argument('--inc_conv_features', type=utils.bool_flag, default=False, help="""Whether or not to incorporate an SSL segmentation via convolutional features on the student""")
    parser.add_argument('--profile', type=utils.bool_flag, default=False, help="""Profile a single call to the model.""")
    return parser


def train_dino(args):
    if args.profile is False:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        to_pil=args.inc_segmentation or args.inc_conv_features,
        target_img_size=[256,192],
    )

    if args.inc_segmentation or args.inc_conv_features:
        dataset = PageLoader(
            docs,
            anchors,
            class_weights,
            fscale,
            config['seg_classes'],
            config['box_classes'],
            batch_size=2,
            num_workers=10,
            synthesis_v2=False,
            df_nodes=None,
            batch_augmentation_count=1,
            dino_transform=transform,
            augmentation=False,
            **data_loader_args,
        )
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transform)

    if args.profile is False:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    # ============ building student and teacher networks ... ============
    # if the network is a vision transformer (i.e. deit_tiny, deit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            include_segmap=args.inc_segmentation,
            use_segmap=args.inc_segmentation,
            include_conv_feature_space=args.inc_conv_features,
            use_conv_feature_space=args.inc_conv_features,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, include_segmap=args.inc_segmentation, use_segmap=False, include_conv_feature_space=args.inc_conv_features, use_conv_feature_space=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        use_segmap=args.inc_segmentation,
        use_conv_features=args.inc_conv_features,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    if args.device.lower() == 'cuda':
        student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student) and args.profile is False:
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    if args.profile is False:
        student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict())
    else:
        teacher_without_ddp.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    )

    if args.profile is False:
        dino_loss = dino_loss.cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        profile=args.profile,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        if args.profile is False:
            data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    loss_func = torch.nn.MSELoss()

    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
    # for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    # for it, (inputs, segmaps, weights, boxes) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        if args.inc_segmentation or args.inc_conv_features:
            # segmentation SSL
            # convert inputs to PIL RGB image in BW
            # Collator is smart enough to return an extra dimension BEFORE the batch dimension when the dataset returns a list of items for one sample

            # For each of IMAGES, SEGMAPS, and WEIGHTS:
            # DATASET returns a LIST of tensors for each batch item of length NCROPS,
            # but the DATALOADER REVERSES the order of items it returns
            # and returns a LIST of length NCROPS,
            # with each item of the list being a TENSOR of length BS.
            # (boxes has not been processed properly anywhere and is ignored).
            # Ignoring whether list or tensor, the shape of images, etc. is therefore
            # (channels is 3 for images, 4 for segmaps (one per segmentation class, inc. none), and 1 for weights):
            # ncrops, batch size, channels, image height, image width
            images, segmaps, weights, boxes = data

            if args.profile is False:
                segmaps = [sm.cuda(non_blocking=True) for sm in segmaps]
                weights = [w.cuda(non_blocking=True) for w in weights]
        else:
            # Collator is smart enough to return an extra dimension BEFORE the batch dimension when the dataset returns a list of items for one sample

            # images:
            # ncrops, batch size, 3, image height, image width
            images, _ = data

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        if args.profile is False:
            images = [im.cuda(non_blocking=True) for im in images]

        DebugLabels = False

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            # Note: Because it is a LIST of tensors passed to the forward() function,
            # PyTorch is smart enough to just increase the batch size accordingly
            ##################################
            # INPUT: NCROPS, batch_size, image
            # (see note above)
            ##################################

            if args.profile:
                with profiler.profile(record_shapes=True) as prof_teacher:
                    teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                with profiler.profile(record_shapes=True) as prof_student:
                    student_output = student(images)
                print(f'******************')
                print(f'TEACHER, self')
                print(prof_teacher.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))
                print(f'******************')
                print('')
                print('')
                print(f'******************')
                print(f'STUDENT, self')
                print(prof_student.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))
                print(f'******************')
                print('')
                print('')
                print(f'******************')
                print(f'TEACHER, total')
                print(prof_teacher.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                print(f'******************')
                print('')
                print('')
                print(f'******************')
                print(f'STUDENT, total')
                print(prof_student.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                print(f'******************')
                assert False, f'Ending execution after profiling a single pass'
            else:
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)

            # segmentation SSL
            ncrops = len(images)
            n_segmaps = 4 # trickiness with parallelization and return values in unpacked lists - for now, hardcode
            if args.inc_segmentation:

                ##################################
                # output for segmaps_:
                # for segmap_ in segmaps_: # one segmap_ per segmentation class: none, chart, list, table
                #   segmap_.shape == [batch_size * NCROPS, image_channels, image_height, image_width], with image_channels == 1 for each class segmentation map
                #    ... data order for first dimension is:
                #   [BS_ncrop_1, BS_ncrop_2, ...]
                ##################################
                student_output, segmaps_0, segmaps_1, segmaps_2, segmaps_3 = student_output

                bs = len(images[0])
                segmaps_ = []
                # One per segmentation class
                # for segmap_ in segmaps_:

                # First dimension was passed as ncrops by dataset, second dimension is this process's batch size in DDP

                # segmap_ arrives as list because it was passed as list
                # The following is the equivalent of: segmap_ = segmap_.chunk(ncrops)
                # ... resulting in a LIST of length 'ncrops', with each element having length 'batch size'
                # segmap_ = [segmap_[b*bs:(b+1*bs)] for b in range(ncrops)]

                segmap_ = [segmaps_0[crop_idx*bs:(crop_idx+1)*bs] for crop_idx in range(ncrops)]
                segmaps_.append(segmap_)
                segmap_ = [segmaps_1[crop_idx*bs:(crop_idx+1)*bs] for crop_idx in range(ncrops)]
                segmaps_.append(segmap_)
                segmap_ = [segmaps_2[crop_idx*bs:(crop_idx+1)*bs] for crop_idx in range(ncrops)]
                segmaps_.append(segmap_)
                segmap_ = [segmaps_3[crop_idx*bs:(crop_idx+1)*bs] for crop_idx in range(ncrops)]
                segmaps_.append(segmap_)

                # segmaps_ = segmaps_tmp_
            elif args.inc_conv_features:
                student_output, segmaps_ = student_output
                segmaps_ = segmaps_.chunk(ncrops)

            loss = dino_loss(student_output, teacher_output, epoch)
            if DebugLabels:
                print(f'loss: {loss}')

            if args.inc_segmentation or args.inc_conv_features:
                # segmentation SSL
                lambda_seg = 2.
                ncrops = len(segmaps)
                seg_loss = None
                for idx in range(ncrops):
                    bs = len(segmaps[idx])
                    for bidx in range(bs):
                        for seg_class_idx in range(n_segmaps):
                            if args.inc_segmentation:
                                if seg_loss is None:
                                    seg_loss  = lambda_seg * loss_func(weights[idx][bidx] * segmaps_[seg_class_idx][idx][bidx], weights[idx][bidx] * segmaps[idx][bidx][seg_class_idx])
                                else:
                                    seg_loss += lambda_seg * loss_func(weights[idx][bidx] * segmaps_[seg_class_idx][idx][bidx], weights[idx][bidx] * segmaps[idx][bidx][seg_class_idx])
                            elif args.inc_conv_features:
                                if seg_loss is None:
                                    seg_loss  = lambda_seg * loss_func(weights[idx][bidx] * segmaps_[idx][bidx][seg_class_idx], weights[idx][bidx] * segmaps[idx][bidx][seg_class_idx])
                                else:
                                    seg_loss += lambda_seg * loss_func(weights[idx][bidx] * segmaps_[idx][bidx][seg_class_idx], weights[idx][bidx] * segmaps[idx][bidx][seg_class_idx])
                        if DebugLabels:
                            print(f'seg_loss: {seg_loss}')
                            pil_img = transforms.ToPILImage()(images[idx][bidx])
                            pil_img.save(f'/data/deepink/image_c{idx}_b{bidx}.png')
                            pil_img = transforms.ToPILImage()(weights[idx][bidx])
                            pil_img.save(f'/data/deepink/weights_c{idx}_b{bidx}.png')
                            for seg_class_idx in range(n_segmaps):
                                pil_img = transforms.ToPILImage()(segmaps[idx][bidx][seg_class_idx])
                                pil_img.save(f'/data/deepink/segmaps_labels_c{idx}_b{bidx}_ch{seg_class_idx}.png')
                                pil_img = transforms.ToPILImage()(segmaps_[seg_class_idx][idx][bidx])
                                pil_img.save(f'/data/deepink/segmaps_preds_c{idx}_b{bidx}_ch{seg_class_idx}.png')
                total_loss = loss + seg_loss
            else:
                total_loss = loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)







        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            total_loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(total_loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            if args.profile is False:
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            else:
                for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if args.profile is False:
            torch.cuda.synchronize()
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(seg_loss=seg_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, to_pil=False, target_img_size=None):
        self.ratio = (0.75, 1.3333333333333333)
        self.to_pil = to_pil
        flip_and_color_jitter = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            # transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.target_img_size = target_img_size

    def __call__(self, image, image2=None, image3=None):

        crops = []
        crops_seg = []
        crops_seg_weights = []

        # WARNING: RANDOM HORIZONTAL FLIP has been LOST from flip_and_color_jitter

        h, w = self.target_img_size[0], self.target_img_size[1]
        # Do not include channels
        # sz1 = (1, h, w)
        # sz3 = (3, h, w)
        sz1 = (h, w)
        sz3 = (h, w)

        if self.to_pil:
            # print(f'A: image.shape: {image.shape}, mean = {torch.mean(torch.abs(image))}')
            image = transforms.ToPILImage()(image).convert("RGB")
            if self.target_img_size is not None:
                image = image.resize((self.target_img_size[1], self.target_img_size[0]))

        crop_params = transforms.RandomResizedCrop.get_params(image, scale=self.global_crops_scale, ratio=self.ratio)
        out = transforms.functional.crop(image, *crop_params)
        # PIL image -
        # Conversion to tensor happens as you can see in the actual transform sequence above
        # print(f'***********************************************\ntype(out): {type(out)}\n***********************************************')
        out = self.global_transfo1(out)
        # print(f'***********************************************\nout.shape: {out.shape}\n***********************************************')
        out = F.interpolate(out.unsqueeze(0), size=sz3).squeeze(0)
        crops.append(out)
        if image2 is not None:
            segmented_reconstructed = torch.tensor(())
            for idx, image_ in enumerate(image2):
                image_ = transforms.functional.crop(image_.unsqueeze(0), *crop_params)
                # image_ = transforms.ToTensor()(image_)
                image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
                segmented_reconstructed = torch.cat([segmented_reconstructed, image_])
            crops_seg.append(segmented_reconstructed)
        if image3 is not None:
            image_ = transforms.functional.crop(image3, *crop_params)
            # image_ = transforms.ToTensor()(image_)
            image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
            crops_seg_weights.append(image_)

        crop_params = transforms.RandomResizedCrop.get_params(image, scale=self.global_crops_scale, ratio=self.ratio)
        out = transforms.functional.crop(image, *crop_params)
        out=self.global_transfo2(out)
        out = F.interpolate(out.unsqueeze(0), size=sz3).squeeze(0)
        crops.append(out)
        if image2 is not None:
            segmented_reconstructed = torch.tensor(())
            for idx, image_ in enumerate(image2):
                image_ = transforms.functional.crop(image_.unsqueeze(0), *crop_params)
                # image_ = transforms.ToTensor()(image_)
                image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
                segmented_reconstructed = torch.cat([segmented_reconstructed, image_])
            crops_seg.append(segmented_reconstructed)
        if image3 is not None:
            image_ = transforms.functional.crop(image3, *crop_params)
            # image_ = transforms.ToTensor()(image_)
            image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
            crops_seg_weights.append(image_)

        for _ in range(self.local_crops_number):
            crop_params = transforms.RandomResizedCrop.get_params(image, scale=self.global_crops_scale, ratio=self.ratio)
            out = transforms.functional.crop(image, *crop_params)
            out=self.local_transfo(out)
            out = F.interpolate(out.unsqueeze(0), size=sz3).squeeze(0)
            crops.append(out)
            if image2 is not None:
                segmented_reconstructed = torch.tensor(())
                for idx, image_ in enumerate(image2):
                    image_ = transforms.functional.crop(image_.unsqueeze(0), *crop_params)
                    # image_ = transforms.ToTensor()(image_)
                    image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
                    segmented_reconstructed = torch.cat([segmented_reconstructed, image_])
                crops_seg.append(segmented_reconstructed)
            if image3 is not None:
                image_ = transforms.functional.crop(image3, *crop_params)
                # image_ = transforms.ToTensor()(image_)
                image_ = F.interpolate(image_.unsqueeze(0), size=sz1).squeeze(0)
                crops_seg_weights.append(image_)

        ######################################################
        # CONVERSION TO TENSOR via transforms SWAPS CWH to CHW
        ######################################################

        # if self.target_img_size is not None:
        #     for i, crop in enumerate(crops):
        #         if crop.size(-2) <= 255:
        #             print(f'{i}: crop.shape: {crop.shape}')
        #     for i, crop_seg in enumerate(crops_seg):
        #         if crop_seg.size(-2) <= 255:
        #             print(f'{i}: crop_seg.shape: {crop_seg.shape}')
        #     for i, crop_seg_weight in enumerate(crops_seg_weights):
        #         if crop_seg_weight.size(-2) <= 255:
        #             print(f'{i}: crop_seg_weight.shape: {crop_seg_weight.shape}')

        if image2 is not None:
            assert image3 is not None
            return crops, crops_seg, crops_seg_weights
        else:
            return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
