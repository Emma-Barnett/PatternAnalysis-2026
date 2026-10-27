import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two 3x3 (2D) or 3x3x3 (3D) convolutions, each followed by batch norm and ReLU."""
    def __init__(self, in_ch, out_ch, dims=2):
        super().__init__()
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d
        Norm = nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            Conv(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
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

class UNet3D(nn.Module):
    """
    3D U-Net (Cicek et al., 2016) with the same four-level topology as UNet2D,
    but with volumetric 3x3x3 convolutions, 2x2x2 max pooling, 2x2x2 transposed
    convolutions and skip connections that carry whole feature volumes.

    Input:  (B, in_ch, X, Y, Z) with X, Y and Z divisible by 16.
    Output: (B, num_classes, X, Y, Z) raw logits.
    """

    def __init__(self, num_classes, in_ch=1, base=16):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_ch, base, dims=3)
        self.down2 = DoubleConv(base, base * 2, dims=3)
        self.down3 = DoubleConv(base * 2, base * 4, dims=3)
        self.down4 = DoubleConv(base * 4, base * 8, dims=3)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16, dims=3)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8, dims=3)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4, dims=3)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2, dims=3)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base, dims=3)

        self.out_conv = nn.Conv3d(base, num_classes, 1)

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

class ContextModule(nn.Module):
    """Pre-activation residual block (Isensee et al., 2018): two 3x3x3
    convolutions with dropout between them, each preceded by instance norm and
    LeakyReLU. The input is added back to the output (residual connection)."""

    def __init__(self, ch, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout3d(p_drop),
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)

class UpsamplingModule(nn.Module):
    """Upscale by 2, then a 3x3x3 convolution that halves the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class LocalizationModule(nn.Module):
    """3x3x3 convolution on the concatenated features, then a 1x1x1
    convolution that halves the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ImprovedUNet3D(nn.Module):
    """
    3D Improved U-Net (Isensee et al., 2018): residual context modules,
    strided-convolution downsampling, localization modules, instance norm,
    LeakyReLU and deep supervision (segmentation maps from three decoder
    levels are upsampled and summed into one output).

    Input:  (B, in_ch, X, Y, Z) with X, Y and Z divisible by 16.
    Output: (B, num_classes, X, Y, Z) raw logits.
    """

    def __init__(self, num_classes, in_ch=1, base=16, p_drop=0.1):
        super().__init__()
        b = base
        # Context (encoder) pathway: conv (stride 2 except level 1) + context module.
        self.conv1 = nn.Conv3d(in_ch, b, 3, padding=1, bias=False)
        self.ctx1 = ContextModule(b, p_drop)
        self.conv2 = nn.Conv3d(b, b * 2, 3, stride=2, padding=1, bias=False)
        self.ctx2 = ContextModule(b * 2, p_drop)
        self.conv3 = nn.Conv3d(b * 2, b * 4, 3, stride=2, padding=1, bias=False)
        self.ctx3 = ContextModule(b * 4, p_drop)
        self.conv4 = nn.Conv3d(b * 4, b * 8, 3, stride=2, padding=1, bias=False)
        self.ctx4 = ContextModule(b * 8, p_drop)
        self.conv5 = nn.Conv3d(b * 8, b * 16, 3, stride=2, padding=1, bias=False)
        self.ctx5 = ContextModule(b * 16, p_drop)

        # Localization (decoder) pathway.
        self.up4 = UpsamplingModule(b * 16, b * 8)
        self.loc4 = LocalizationModule(b * 16, b * 8)
        self.up3 = UpsamplingModule(b * 8, b * 4)
        self.loc3 = LocalizationModule(b * 8, b * 4)
        self.up2 = UpsamplingModule(b * 4, b * 2)
        self.loc2 = LocalizationModule(b * 4, b * 2)
        self.up1 = UpsamplingModule(b * 2, b)
        self.final = nn.Sequential(
            nn.Conv3d(b * 2, b * 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(b * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Deep supervision: 1x1x1 segmentation layers at 1/4, 1/2 and full resolution.
        self.seg3 = nn.Conv3d(b * 4, num_classes, 1)
        self.seg2 = nn.Conv3d(b * 2, num_classes, 1)
        self.seg1 = nn.Conv3d(b * 2, num_classes, 1)
        self.upscale = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x):
        s1 = self.ctx1(self.conv1(x))    # b channels,   full resolution
        s2 = self.ctx2(self.conv2(s1))   # 2b,           1/2
        s3 = self.ctx3(self.conv3(s2))   # 4b,           1/4
        s4 = self.ctx4(self.conv4(s3))   # 8b,           1/8
        x = self.ctx5(self.conv5(s4))    # 16b,          1/16 (bottleneck)

        x = self.loc4(torch.cat([self.up4(x), s4], dim=1))       # 8b, 1/8
        l3 = self.loc3(torch.cat([self.up3(x), s3], dim=1))      # 4b, 1/4
        l2 = self.loc2(torch.cat([self.up2(l3), s2], dim=1))     # 2b, 1/2
        out = self.final(torch.cat([self.up1(l2), s1], dim=1))   # 2b, full

        # Sum the coarse segmentation maps into the finer ones.
        s3 = self.seg3(l3)                       # 1/4 resolution
        s2 = self.seg2(l2)                       # 1/2 resolution
        seg = self.upscale(s3) + s2
        seg = self.upscale(seg) + self.seg1(out)
        if self.training:                        # auxiliary outputs for deep supervision
            size = seg.shape[2:]
            aux2 = F.interpolate(s2, size=size, mode="trilinear", align_corners=False)
            aux3 = F.interpolate(s3, size=size, mode="trilinear", align_corners=False)
            return seg, (aux2, aux3)
        return seg

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

MODELS = {
    "unet2d": UNet2D,
    "unet3d": UNet3D,
    "improved_unet3d": ImprovedUNet3D,
}

def build_model(name, num_classes, base=64):
    """Create a model by name. New models get registered in MODELS."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name](num_classes, in_ch=1, base=base)