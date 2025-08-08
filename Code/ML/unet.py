# -----------------------------------------------------------------------------
# File          : unet.py
# Description   : Daniel Li implementation of UNet
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import cat


# U-Net
class UNet(nn.Module):
    def __init__(self, inChans=1, classes=1, depth=5):
        super().__init__()
        
        self.depth = depth

        channels = [64 * (2**i) for i in range(depth)]
        channels = [inChans] + channels + channels[::-1][1:]

        self.blocks = nn.ModuleList([
            self.block(inChan, outChan) for inChan, outChan in
            zip(channels[:-1], channels[1:])
        ])

        self.upConvs = nn.ModuleList([
            nn.ConvTranspose2d(chan * 2, chan, kernel_size=2, stride=2)
            for chan in [64 * (2**i) for i in range(depth-1)][::-1]
        ])

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lastConv = nn.Conv2d(64, classes, kernel_size=1)

    def block(self, inChans, outChans):
        return nn.Sequential(
            nn.Conv2d(inChans, outChans, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChans, outChans, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outs = []
        for i in range(self.depth-1):
            x = self.blocks[i](x)
            outs.append(x)
            x = self.maxPool(x)

        x = self.blocks[self.depth-1](x)
        outs = outs[::-1]

        for i in range(self.depth-1):
            x = self.upConvs[i](x)
            x = cat([outs[i], x], dim=1)
            x = self.blocks[self.depth+i](x)
        
        x = self.lastConv(x)

        return x
