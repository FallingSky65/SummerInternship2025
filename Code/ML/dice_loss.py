# -----------------------------------------------------------------------------
# File          : dice_loss.py
# Description   : dice loss criterion
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits, shape (B, 1, H, W)
        # targets: binary mask, shape (B, 1, H, W)

        inputs = torch.sigmoid(inputs)

        inputs, targets = inputs.view(-1), targets.view(-1)

        TP = (inputs * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = ((1 - inputs) * targets).sum()

        dice = (2 * TP + self.smooth) / (2 * TP + FP + FN)

        return 1 - dice
