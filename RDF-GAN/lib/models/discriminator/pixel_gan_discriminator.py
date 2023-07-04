import torch.nn as nn


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminaor (pixelGAN)"""

    def __init__(self, in_channels, ndf=64):
        super(PixelDiscriminator, self).__init__()
        net = [
            nn.Conv2d(in_channels, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf * 2, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
