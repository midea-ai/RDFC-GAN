import torch
import torch.nn as nn
from math import sqrt
import warnings


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
            # UserWarning: Mixed memory format inputs detected while calling the operator. The operator will output contiguous tensor even if some of the inputs are in channels_last format.
            # (Triggered internally at  /pytorch/aten/src/ATen/native/TensorIterator.cpp:918.)
            out = gamma * out + beta
        else:
            gamma_weight = self.gamma_weight_layer(input)
            # gamma_weight = self.gamma_weight_layer(style)
            beta_weight = self.beta_weight_layer(input)
            out = gamma_weight * gamma * out + beta_weight * beta

        return out

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        out = normalized_feat * style_std.expand(size) + style_mean.expand(size)

        return out


class IN(nn.Module):
    def __init__(self,in_channel, style_dim):
        super(IN, self).__init__()
        self.down_channel = nn.Conv2d(in_channel + style_dim,in_channel,(1,1))
        self.norm = nn.InstanceNorm2d(in_channel + style_dim)

    def forward(self, content_feat, style_feat):
        out = torch.cat([content_feat,style_feat],dim=1)
        out = self.norm(out)
        out = self.down_channel(out)
        return out

class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer_type=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        warnings.warn('parameters groups, base_width and norm_layer are '
                      'ignored in NonBottleneck1D')
        if norm_layer_type is None or norm_layer_type == 'BN2d':
            # use BatchNorm2d by default
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'IN2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError(f'Only BN2d and IN2d are supported so far'
                                      f'but got {norm_layer_type}')
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = norm_layer(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = norm_layer(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + identity)


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


class DecoderModule(nn.Module):
    """For GAN, use LeakyReLU may be better"""
    def __init__(self,
                 channels_in,
                 channels_out,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=0,
                 norm_layer_type=None,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear'):
        super(DecoderModule, self).__init__()
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
