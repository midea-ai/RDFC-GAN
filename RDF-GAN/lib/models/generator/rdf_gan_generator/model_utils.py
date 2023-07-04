import torch
import torch.nn as nn
from math import sqrt
from lib.models.backbone.resnet import NonBottleneck1D


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EqualLinear, self).__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim, weighting=False):
        super(AdaptiveInstanceNorm, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

        if weighting:
            self.gamma_weight_layer = nn.Conv2d(in_channel, in_channel, (1, 1))
            # self.gamma_weight_layer = nn.Conv2d(style_dim, style_dim, (1, 1))
            self.beta_weight_layer = nn.Conv2d(in_channel, in_channel, (1, 1))
            self.weighting = True
        else:
            self.weighting = False


    def forward(self, input, style):
        # style = self.style(style).unsqueeze(2).unsqueeze(3)
        style = style.permute(0, 2, 3, 1)
        style = self.style(style).permute(0, 3, 1, 2)   # (B, C, H, W)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)

        if not self.weighting:
            out = gamma * out + beta
        else:
            gamma_weight = self.gamma_weight_layer(input)
            # gamma_weight = self.gamma_weight_layer(style)
            beta_weight = self.beta_weight_layer(input)
            out = gamma_weight * gamma * out + beta_weight * beta

        return out


class ConvNormAct(nn.Sequential):
    """A Sequential to perform conv + norm + act, 2D version"""

    def __init__(self, channels_in, channels_out, kernel_size, norm_layer_type=None,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvNormAct, self).__init__()
        if norm_layer_type == None or norm_layer_type == 'BN2d':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm_layer_type == 'IN2d':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError(f'Only BN2d and IN2d are supported so far'
                                      f'but got {norm_layer_type}')
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in,
                                          channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=use_bias,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('norm', norm_layer(channels_out))
        self.add_module('act', activation)


class Upsampling(nn.Module):
    """Perform 2x upsampling"""

    def __init__(self, mode, channels=None):
        super(Upsampling, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = True

        if 'learned-3x3' in mode:
            if mode == 'learned-3x3':
                self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))
            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x, upper_size=None):

        if upper_size is not None:
            size = upper_size
        else:
            # 2x upsample by default.
            size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        # import pdb
        # pdb.set_trace()
        # size = (int(x.shape[2]*2 + 1), int(x.shape[3]*2) + 1)
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


class DCVGANDecoderModule(nn.Module):
    """For GAN, use LeakyReLU may be better"""
    def __init__(self,
                 channels_in,
                 channels_out,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=0,
                 norm_layer_type=None,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear'):
        super(DCVGANDecoderModule, self).__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvNormAct(channels_in,
                                   channels_out,
                                   kernel_size=3,
                                   norm_layer_type=norm_layer_type,
                                   activation=activation)
        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_out,
                                          channels_out,
                                          norm_layer_type=norm_layer_type,
                                          activation=activation))
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsampling(mode=upsampling_mode,
                                   channels=channels_out)

    def forward(self, decoder_features, encoder_features=None, up_size=None):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if encoder_features is not None:
            assert up_size is None
            up_size = tuple(encoder_features.shape[-2:])

        out = self.upsample(out,
                            up_size)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out


if __name__ == '__main__':
    model = AdaptiveInstanceNorm(in_channel=128, style_dim=64)

    # for latent code case, use `style = self.style(style).unsqueeze(2).unsqueeze(3)`
    # for feature map case, use `style = self.style(style).permute(0, 3, 1, 2)`
    # or convert the feature map to latent code format, parameters ars massive
    # dummy_depth = torch.randn((1, 64))       # latent code
    dummy_depth = torch.randn((1, 16, 16, 64))
    dummy_rgb = torch.randn((1, 128, 16, 16))

    out = model(dummy_rgb, dummy_depth)
