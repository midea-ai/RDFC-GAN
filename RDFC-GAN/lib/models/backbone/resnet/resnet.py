import os
import torch
import torch.nn as nn
import warnings
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer_type=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super(BasicBlock, self).__init__()
        self.residual_only = residual_only
        if norm_layer_type is None or norm_layer_type == 'BN2d':
            # use BatchNorm2d by default
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'IN2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError(f'Only BN2d and IN2d are supported so far'
                                      f'but got {norm_layer_type}')
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and '
                             'base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.act = activation
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual_only:
            return out
        out = out + identity
        out = self.act(out)

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer_type=None,
                 activation=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if norm_layer_type is None or norm_layer_type == 'BN2d':
            # use BatchNorm2d by default
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'IN2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError(f'Only BN2d and IN2d are supported so far'
                                      f'but got {norm_layer_type}')
        width = int(planes * (base_width / 64.)) * groups
        # both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, block, zero_init_residual=False, groups=1,
                 with_per_group=64, replace_stride_with_dilation=None,
                 dilation=None, norm_layer_type=None, input_channels=3, activation=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        self.norm_layer_type = norm_layer_type
        if norm_layer_type is None or norm_layer_type == 'BN2d':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'IN2d':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        self.norm_layer = norm_layer

        self.inplanes = 64        # channels that fed into start resnet block
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None'
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))

        if dilation is not  None:
            dilation = dilation
            if len(dilation) != 4:
                raise ValueError('dilation should be None or a 4-element tuple, got {}'.format(dilation))
        else:
            dilation = [1, 1, 1, 1]

        self.groups = groups
        self.base_width = with_per_group

        # down_2
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act = activation
        # down_4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down_2_channels_out = 64
        if self.replace_stride_with_dilation == [False, False, False]:
            # origin resnet
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 128 * block.expansion
            self.down_16_channels_out = 256 * block.expansion
            self.down_32_channels_out = 512 * block.expansion
        elif self.replace_stride_with_dilation == [False, True, True]:
            self.down_8_channels_out = 64 * block.expansion
            self.down_16_channels_out = 512 * block.expansion

        self.layer1 = self._make_layer(block, 64, layers[0], dilate=dilation[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=dilation[1],
                                       replace_stride_with_dilation=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=dilation[2],
                                       replace_stride_with_dilation=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=dilation[3],
                                       replace_stride_with_dilation=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual
            # block behaves like an identity. This improves the model by 0.2~0.3%
            # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, block_num, stride=1, dilate=1, replace_stride_with_dilation=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if replace_stride_with_dilation:
            self.dilation *= stride
            stride = 1
        if dilate > 1:
            self.dilation = dilate
            dilate_first_block = previous_dilation
        else:
            dilate_first_block = previous_dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilate_first_block, self.norm_layer_type,
                            activation=self.act))
        self.inplanes = planes * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer_type=self.norm_layer_type,
                                activation=self.act))

        return nn.Sequential(*layers)

    def forward_first_conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        return x

    def forward_layer1(self, x):
        x = self.forward_resblock(x, self.layer1)
        self.skip1_channels = x.size()[1]
        return x

    def forward_layer2(self, x):
        x = self.forward_resblock(x, self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x, self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock(x, self.layer4)
        return x


    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x


    def forward(self, _input):
        x = self.conv1(_input)
        x = self.bn1(x)
        x_down2 = self.act(x)
        x = self.maxpool(x_down2)

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)

        if self.replace_stride_with_dilation == [False, False, False]:
            features = [x_layer4, x_layer3, x_layer2, x_layer1]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
        elif self.replace_stride_with_dilation == [False, True, True]:
            # x has resolution 1/8
            # skip4 has resolution 1/8
            # skip3 has resolution 1/8
            # skip2 has resolution 1/8
            # skip1 has resolution 1/4
            # x_down2 has resolution 1/2
            features = [x, x_layer1, x_down2]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]

        return features


def ResNet18(pretrained_on_imagenet=False,
             pretrained_dir='./pretrained_model/resnet_on_imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            kwargs['block'] = globals()[kwargs['block']]
        else:
            raise NotImplementedError
    model = ResNet(layers=[2, 2, 2, 2], **kwargs)

    if pretrained_on_imagenet:
        ckpt_path = os.path.join(pretrained_dir, model_urls['resnet18'].split('/')[-1])
        if os.path.exists(ckpt_path):
            weights = torch.load(ckpt_path)
        else:
            weights = model_zoo.load_url(model_urls['resnet18'], model_dir=pretrained_dir)
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'], dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Load ResNet18 pretrained on ImageNet')

    return model


def ResNet34(pretrained_on_imagenet=False,
             pretrained_dir='./pretrained_model/resnet_on_imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            kwargs['block'] = globals()[kwargs['block']]
        else:
            raise NotImplementedError
    model = ResNet(layers=[3, 4, 6, 3], **kwargs)

    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3

    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r34',
            pretrained_dir=pretrained_dir)
    elif pretrained_on_imagenet:
        ckpt_path = os.path.join(pretrained_dir, model_urls['resnet34'].split('/')[-1])
        if os.path.exists(ckpt_path):
            weights = torch.load(ckpt_path)
        else:
            weights = model_zoo.load_url(model_urls['resnet34'], model_dir=pretrained_dir)
        if input_channels == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'], dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Load ResNet34 pretrained on ImageNet')

    return model


def ResNet50(pretrained_on_imagenet=False,
             pretrained_dir='./pretrained_model/resnet_on_imagenet',
             **kwargs):
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   **kwargs)
    if pretrained_on_imagenet:
        ckpt_path = os.path.join(pretrained_dir, model_urls['resnet50'].split('/')[-1])
        if os.path.exists(ckpt_path):
            weights = torch.load(ckpt_path)
        else:
            weights = model_zoo.load_url(model_urls['resnet50'], model_dir=pretrained_dir)
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'], dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Load ResNet50 pretrained on ImageNet')

    return model


def load_pretrained_with_different_encoder_block(model, encoder_block, input_channels, resnet_name,
                                                 pretrained_dir='./pretrained_model/resnet_on_imagenet'):
    import os
    from collections import OrderedDict
    ckpt_path = os.path.join(pretrained_dir, f'{resnet_name}_NBt1D.pth')

    checkpoint = torch.load(ckpt_path)

    checkpoint['state_dict2'] = OrderedDict()

    # rename keys and leave out last fully connected layer
    for key in checkpoint['state_dict']:
        if 'encoder' in key:
            checkpoint['state_dict2'][key.split('encoder.')[-1]] = \
                checkpoint['state_dict'][key]
    weights = checkpoint['state_dict2']

    if input_channels == 1:
        # sum the weights of the first convolution
        weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                            axis=1,
                                            keepdim=True)

    model.load_state_dict(weights, strict=False)
    print(f'Loaded {resnet_name} with encoder block {encoder_block} '
          f'pretrained on ImageNet')
    print(ckpt_path)
    return model



if __name__ == '__main__':
    model = ResNet50(pretrained_on_imagenet=True, pretrained_dir='../../../../pretrained_model/resnet_on_imagenet')

    print(model)

    model.eval()
    image = torch.randn((1, 3, 224, 224))

    with torch.no_grad():
        outputs = model(image)
    for t in outputs:
        print(t.shape)
