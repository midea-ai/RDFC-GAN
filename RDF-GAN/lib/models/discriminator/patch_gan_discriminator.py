import torch.nn as nn
import warnings
from lib.models.module.deprecated import ConvModule
warnings.warn('ConvModule maybe deprecated in the future')


class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator.

    refers from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538
    """
    def __init__(self,
                 in_channels,
                 out_channels=(64, 128, 256, 512, 1),
                 kernel_size=(4, 4, 4, 4, 4),
                 stride=(2, 2, 2, 1, 1),
                 padding=(1, 1, 1, 1, 1),
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN', in_discriminator=True),
                 activation='ReLU'):
        super(PatchGANDiscriminator, self).__init__()
        assert out_channels[-1] == 1, f"The channel of the feature map obtained by the last conv layer" \
                                      f"must be 1, but got {out_channels[-1]}"
        conv_channels = [in_channels] + list(out_channels)
        sequence = []

        num_convs = len(conv_channels) - 1
        for i in range(len(conv_channels) - 1):
            sequence.append(ConvModule(in_channels=conv_channels[i],
                                       out_channels=conv_channels[i + 1],
                                       kernel_size=kernel_size[i],
                                       stride=stride[i],
                                       padding=padding[i],
                                       conv_cfg=conv_cfg,
                                       norm_cfg=None if i == 0 or i == num_convs - 1 else norm_cfg,
                                       activation=None if i == num_convs - 1 else activation))
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
