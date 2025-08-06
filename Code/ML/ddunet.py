# -----------------------------------------------------------------------------
# File          : ddunet.py
# Description   : Daniel Li implementation of Dense Dense U-Net based on
#                 https://ieeexplore.ieee.org/document/9607593
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import cat
from pytorch_wavelets import DWTForward, DWTInverse


# local residual dense block
# 64 channels in and out (B, 64, 320, 320)
# 3 blocks:
# 1:
#   in(64): z0(64)
#   out(32): z1(32)
# 2:
#   in(96): conv(z0)(64) + z1(32)
#   out(32): z2(32)
# 3:
#   in(128): conv(z0)(64) + z1(32) + z2(32)
#   out(64): zLF(64)
class LRDB(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

        self.block1 = nn.Sequential(
            nn.Conv2d(ch, ch//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(ch//2, ch//2, 3, 1, 1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(3*ch//2, ch//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(ch//2, ch//2, 3, 1, 1),
            nn.ReLU()
        )

        self.block3 = nn.Conv2d(ch*2, ch, 3, 1, 1)

    def forward(self, x):
        z0 = x
        conv_z0_1 = self.conv1(z0)
        conv_z0_2 = self.conv2(z0)
        z1 = self.block1(z0)
        z2 = self.block2(cat([conv_z0_1, z1], dim=1))
        zLF = self.block3(cat([conv_z0_2, z1, z2], dim=1))
        zOut = z0 + zLF
        return zOut


# dense U-Net block
# 7 blocks
# 1:
#   in(64 * n): x0_0, ... , x0_n-1
#   out(64): y0
# 2:
#   in(64 * n): DWT(y0), x1_1, ..., x1_n-1
#   out(64): y1
# 3:
#   in(64 * n): DWT(y1), x2_1, ..., x2_n-1
#   out(64): y2
# 4:
#   in(64 * n): DWT(y2), x3_1, ..., x3_n-1
#   out(64): x3_n
# 5:
#   in(128): y_2, IWT(x3_n)
#   out(64): x2_n
# 6:
#   in(128): y_1, IWT(x2_n)
#   out(64): x1_n
# 7:
#   in(128): y_0, IWT(x1_n)
#   out(64): x0_n
class DUNetB(nn.Module):
    def __init__(self, n, ch=64):
        super().__init__()
        self.n = n

        self.block1 = self.block(ch * n, ch)
        self.block2 = self.block(ch * n, ch)
        self.block3 = self.block(ch * n, ch)

        self.block4 = nn.Sequential(
            nn.Conv2d(ch * n, ch, 3, 1, 1), nn.ReLU(), LRDB(ch=ch), LRDB(ch=ch))

        self.block5 = self.block(ch*2, ch)
        self.block6 = self.block(ch*2, ch)
        self.block7 = self.block(ch*2, ch)

        self.DWT = DWTForward(J=1, wave='haar', mode='symmetric')
        self.IWT = DWTInverse(wave='haar', mode='symmetric')

    def block(self, inChans, ch=64):
        return nn.Sequential(nn.Conv2d(inChans, ch, 3, 1, 1),
                             nn.ReLU(), LRDB(ch=ch))

    def yh(self, yl):
        shape = list(yl.shape)
        shape.insert(2, 3)
        zeros = torch.zeros(shape, device=yl.device)
        return [zeros]

    # input:
    #   x0 (B, 64 * n, 320, 320)
    #   x1 (B, 64 * (n-1), 320, 320)
    #   x2 (B, 64 * (n-1), 320, 320)
    #   x3 (B, 64 * (n-1), 320, 320)
    # output:
    #   x0_n (B, 64, 320, 320)
    #   x1_n (B, 64, 320, 320)
    #   x2_n (B, 64, 320, 320)
    #   x3_n (B, 64, 320, 320)
    def forward(self, x0, x1=None, x2=None, x3=None, last=False):
        y0 = self.block1(x0)
        if self.n == 1:
            y1 = self.block2(self.DWT(y0)[0])
            y2 = self.block3(self.DWT(y1)[0])
            x3_n = self.block4(self.DWT(y2)[0])
        else:
            y1 = self.block2(cat([self.DWT(y0)[0], x1], dim=1))
            y2 = self.block3(cat([self.DWT(y1)[0], x2], dim=1))
            x3_n = self.block4(cat([self.DWT(y2)[0], x3], dim=1))
        x2_n = self.block5(cat([self.IWT([x3_n, self.yh(x3_n)]), y2], dim=1))
        x1_n = self.block5(cat([self.IWT([x2_n, self.yh(x2_n)]), y1], dim=1))
        x0_n = self.block5(cat([self.IWT([x1_n, self.yh(x1_n)]), y0], dim=1))
        if last: # others aren't used in the last DUnet
            return x0_n
        else:
            return x0_n, x1_n, x2_n, x3_n


# dense dense U-Net
# in: image (B, 1, 320, 320)
# out: denoised image (B, 1, 320, 320)
class DDUNet(nn.Module):
    def __init__(self, nChans=1, nblocks=5, ch=64):
        super().__init__()
        self.nblocks = nblocks
        self.preblock = nn.Sequential(
            nn.Conv2d(nChans, ch, 3, 1, 1), nn.ReLU())
        self.DUNetBlocks = nn.ModuleList(
            [DUNetB(i+1, ch=ch) for i in range(nblocks)])
        self.convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, 1, 1) for i in range(nblocks)])
        self.postblock = nn.Sequential(
            nn.Conv2d(ch * (nblocks + 1), ch, 3, 1, 1),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(ch, nChans, 3, 1, 1)
        )

    def forward(self, x):
        x0_0 = self.preblock(x)
        x0_1, x1_1, x2_1, x3_1 = self.DUNetBlocks[0](x0_0)
        x0, x1, x2, x3 = [x0_1], [x1_1], [x2_1], [x3_1]
        for i in range(1, self.nblocks - 1):
            x0_n, x1_n, x2_n, x3_n = self.DUNetBlocks[i](
                cat([self.convs[i-1](x0_0)] + x0, dim=1),
                cat(x1, dim=1), cat(x2, dim=1), cat(x3, dim=1))
            x0.append(x0_n)
            x1.append(x1_n)
            x2.append(x2_n)
            x3.append(x3_n)
        x0_n = self.DUNetBlocks[-1](
                cat([self.convs[-2](x0_0)] + x0, dim=1),
                cat(x1, dim=1), cat(x2, dim=1), cat(x3, dim=1), last=True)
        x0.append(x0_n)
        y = self.postblock(cat([self.convs[-1](x0_0)] + x0, dim=1))
        return x + y
