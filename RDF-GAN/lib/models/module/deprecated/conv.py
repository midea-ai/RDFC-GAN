"""
refers from mmcv
"""
import torch.nn as nn
import warnings
from .init_weights import kaiming_init, normal_init, xavier_init, constant_init
from .activation import act_layers
from .norm import build_norm_layer


class ConvModule(nn.Module):
    """
    only support conv2d currently.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='ReLU',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary
        if bias == 'auto':
            if norm_cfg is None:
                bias = True
            else:
                bias = 'BN' not in norm_cfg.get('type')
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.activation:
                x = self.act(x)
        return x


class DepthWiseConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias='auto',
                 norm_cfg=dict(type='BN'),
                 activation='ReLU',
                 inplace=True,
                 order=('depthwise_conv', 'dwnorm', 'act', 'pointwise_conv', 'pwnorm', 'act')):
        super(DepthWiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == set(['depthwise_conv', 'dwnorm', 'act', 'pointwise_conv', 'pwnorm', 'act'])

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise_conv.stride
        self.padding = self.depthwise_conv.padding
        self.dilation = self.depthwise_conv.dilation
        self.transposed = self.depthwise_conv.transposed
        self.output_padding = self.depthwise_conv.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.depthwise_conv, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise_conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != 'act':
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == 'act' and self.activation:
                x = self.act(x)
        return x
