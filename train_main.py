#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import os
import cv2
import random
import time
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from Network import RAINet
from DerainDataset import TrainDataset

from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# ----------------------
# Arguments
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/magecliff/rainnet/dataset/Rain100L/rain")
parser.add_argument("--gt_path", type=str,
                    default="/home/magecliff/rainnet/dataset/Rain100L/norain")

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=12)
parser.add_argument('--patchSize', type=int, default=64)
parser.add_argument('--niter', type=int, default=100)

parser.add_argument('--num_M', type=int, default=32)
parser.add_argument('--num_Z', type=int, default=32)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--S', type=int, default=20)

parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--milestone', type=int, default=[25, 50, 75])
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--log_dir', default='./checkpoints/Rain100L_single/')
parser.add_argument('--model_dir', default='./checkpoints/Rain100L_single/')

parser.add_argument('--manualSeed', type=int, default=6488)

opt = parser.parse_args()


# ----------------------
# Setup
# ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
device = torch.device("cuda:0")

torch.cuda.empty_cache()

os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True


# ----------------------
# Training loop
# ----------------------
def train_model(net, optimizer, lr_scheduler, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    writer = SummaryWriter(opt.log_dir)
    step = 0
    num_data = len(dataset)
    num_iter_epoch = ceil(num_data / opt.batchSize)

    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        lr = optimizer.param_groups[0]['lr']

        for ii, data in enumerate(data_loader):
            im_rain, im_gt = [x.cuda(non_blocking=True) for x in data]

            net.train()
            optimizer.zero_grad()

            outputs, _ = net(im_rain, labels=im_gt)
            loss = outputs[0]
            out = outputs[-1]

            loss.backward()
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter

            if ii % 300 == 0:
                out = torch.clamp(out, 0., 255.)
                pre_out = out.detach().cpu().numpy().astype(np.uint8)
                gt_np = im_gt.detach().cpu().numpy().astype(np.uint8)

                psnr = []
                for i in range(len(pre_out)):
                    psnr.append(
                        compare_psnr(
                            pre_out[i].transpose(1, 2, 0),
                            gt_np[i].transpose(1, 2, 0),
                            data_range=255
                        )
                    )

                avg_psnr = sum(psnr) / len(psnr)

                print(f"[Epoch {epoch+1}/{opt.niter}] "
                      f"{ii}/{num_iter_epoch} "
                      f"Loss={mse_iter:.2e} "
                      f"PSNR={avg_psnr:.2f} "
                      f"lr={lr:.2e}")

                writer.add_scalar('train/Loss', mse_iter, step)
                writer.add_scalar('train/PSNR', avg_psnr, step)

            step += 1

        mse_per_epoch /= (ii + 1)
        print(f"Epoch {epoch+1} Avg Loss: {mse_per_epoch:.2e}")

        lr_scheduler.step()

        save_path = os.path.join(
            opt.model_dir, f"DerainNet_state_{epoch+1}.pt"
        )
        torch.save(net.state_dict(), save_path)

        print(f"Epoch time: {time.time() - tic:.2f}s")
        print("-" * 80)

    writer.close()
    print("Training finished.")


# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    netDerain = RAINet(opt).cuda()

    macs, params = get_model_complexity_info(
        netDerain, (3, 64, 64),
        as_strings=True,
        flops_units="GMac",
        param_units="M",
        print_per_layer_stat=False
    )

    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")

    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lr)
    schedulerDerain = MultiStepLR(
        optimizerDerain, milestones=opt.milestone, gamma=0.2
    )

    train_dataset = TrainDataset(
        opt.data_path,
        opt.gt_path,
        opt.patchSize,
        opt.batchSize * 1500
    )

    train_model(netDerain, optimizerDerain, schedulerDerain, train_dataset)
