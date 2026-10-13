# Containing the source code of the components of your model. 
# Source code for DoubleConv, UNet2D, DiceLoss, CombinedLoss, and build_model. 

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet2D(nn.Module):
    """
    Standard 2D U-Net (Ronneberger et al., 2015) with four down/up levels.

    Input:  (B, in_ch, H, W) with H and W divisible by 16.
    Output: (B, num_classes, H, W) raw logits.
    """

    def __init__(self, num_classes, in_ch=1, base=64):
        super().__init__()
        # Encoder: halve the spatial size, double the channels at each level.
        self.down1 = DoubleConv(in_ch, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder: upsample, concatenate the matching encoder features, convolve.
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        # One logit per class per pixel.
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        u4 = self.dec4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.dec3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out_conv(u1)


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss, averaged over classes.

    Works for 2D (B, C, H, W) and 3D (B, C, D, H, W) logits. Targets are
    integer class maps without the channel dimension.
    """

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits.float(), dim=1)
        onehot = F.one_hot(targets, self.num_classes)
        onehot = onehot.movedim(-1, 1).float()  # channels-last -> channels-first
        dims = (0,) + tuple(range(2, logits.dim()))  # sum over batch and space
        intersection = torch.sum(probs * onehot, dims)
        cardinality = torch.sum(probs + onehot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Cross-entropy + Dice. Dice counters the heavy class imbalance (the
    prostate is a tiny fraction of the pixels); cross-entropy keeps the
    gradients well behaved early in training."""

    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return self.ce(logits.float(), targets) + self.dice(logits, targets)

def build_model(name, num_classes, base=64):
    """Create a model by name. New models get registered here."""
    if name == "unet2d":
        return UNet2D(num_classes, in_ch=1, base=base)
    raise ValueError(f"Unknown model '{name}'. Available: unet2d")