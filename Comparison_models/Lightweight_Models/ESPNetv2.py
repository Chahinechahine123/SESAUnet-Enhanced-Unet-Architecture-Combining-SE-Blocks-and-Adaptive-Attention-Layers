import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Basic Conv Block
# -------------------------------------------------
class ConvBNPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.block(x)


# -------------------------------------------------
# EESP Block (Core of ESPNetv2)
# -------------------------------------------------
class EESP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, branches=4, r_lim=7):
        super().__init__()

        self.stride = stride
        k = out_channels // branches

        self.proj_1x1 = ConvBNPReLU(in_channels, k * branches, 1)

        self.branches = nn.ModuleList()
        for i in range(branches):
            dilation = min(i + 1, r_lim)
            self.branches.append(
                ConvBNPReLU(k, k, 3, stride=stride, groups=k)
            )

        self.merge = ConvBNPReLU(k * branches, out_channels, 1)

        if stride == 1 and in_channels == out_channels:
            self.use_residual = True
        else:
            self.use_residual = False

    def forward(self, x):
        identity = x

        output = self.proj_1x1(x)
        splits = torch.chunk(output, len(self.branches), dim=1)

        outputs = []
        for s, branch in zip(splits, self.branches):
            outputs.append(branch(s))

        out = torch.cat(outputs, dim=1)
        out = self.merge(out)

        if self.use_residual:
            out = out + identity

        return out


# -------------------------------------------------
# Downsampling Block
# -------------------------------------------------
class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eesp = EESP(in_channels, out_channels, stride=2)
        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.eesp(x)


# -------------------------------------------------
# ESPNetv2 for Segmentation
# -------------------------------------------------
class ESPNetv2_Seg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Encoder
        self.level1 = ConvBNPReLU(3, 32, 3, stride=2)
        self.level2 = DownSampler(32, 64)
        self.level3 = DownSampler(64, 128)
        self.level4 = DownSampler(128, 256)

        self.eesp_blocks = nn.Sequential(
            EESP(256, 256),
            EESP(256, 256),
            EESP(256, 256)
        )

        # Decoder (simple upsampling)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        size = x.size()[2:]

        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        x = self.eesp_blocks(x)

        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x


# -------------------------------------------------
# Test
# -------------------------------------------------
if __name__ == "__main__":
    model = ESPNetv2_Seg(num_classes=1)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print("Output shape:", y.shape)