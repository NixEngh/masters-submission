import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on: https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with adjustable depth.
    depth: number of resolution levels (original architecture = 5). Supported: 3,4,5.
    depth=5 -> original (inc + 4 downs).
    Shallower (4 or 3) removes deepest levels, reducing params and receptive field.
    """

    def __init__(self, in_channels=37, out_channels=1, bilinear=False,
                 depth=5, dropout_p=0.1, **kwargs):
        super().__init__()
        assert 3 <= depth <= 5, "depth must be 3, 4, or 5"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth

        full_channels = [64, 128, 256, 512, 1024]  # canonical progression
        selected = full_channels[:depth]
        factor = 2 if bilinear else 1
        base_ch = selected[0]

        # Initial conv
        self.inc = DoubleConv(in_channels, base_ch)

        # Encoder (Down blocks)
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            in_ch = selected[i - 1]
            out_ch = selected[i]
            if i == depth - 1:
                # Last down block applies factor division like original (1024//2 when bilinear)
                out_ch = out_ch // factor
            self.downs.append(Down(in_ch, out_ch))

        # Collect encoder output channel sizes (after each stage)
        encoder_out_channels = [base_ch]
        for i in range(1, depth):
            ch = selected[i]
            if i == depth - 1:
                ch = ch // factor
            encoder_out_channels.append(ch)

        # Decoder (Up blocks)
        skip_channels = encoder_out_channels[:-1]
        current_channels = encoder_out_channels[-1]
        self.ups = nn.ModuleList()

        for skip_ch in reversed(skip_channels):
            if bilinear:
                in_ch_up = current_channels + skip_ch  # after cat
                # Match original: halve except final (base) stage
                if skip_ch > base_ch:
                    out_ch_up = skip_ch // 2
                else:
                    out_ch_up = skip_ch
            else:
                in_ch_up = current_channels
                out_ch_up = skip_ch
            self.ups.append(Up(in_ch_up, out_ch_up, bilinear))
            current_channels = out_ch_up

        self.outc = OutConv(current_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        bottom = skips[-1]
        skip_feats = skips[:-1]

        x = self.dropout(bottom)  # Apply dropout to bottleneck
        for up, skip in zip(self.ups, reversed(skip_feats)):
            x = up(x, skip)

        return self.outc(x)


