import torch
import torch.nn as nn
import torchvision


model_path = {
    'resnet18': 'pretrained_model/resnet/resnet18.pth',
    'resnet34': 'pretrained_model/resnet/resnet34.pth'
}


def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet18'])
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


def conv_bn_relu(channels_in, channels_out, kernel, stride=1, padding=0, bn=True, _in=False, relu=True):
    assert not (bn and _in)
    layers = []
    layers.append(nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=not bn))

    if bn:
        layers.append(nn.BatchNorm2d(channels_out))
    if _in:
        layers.append(nn.InstanceNorm2d(channels_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers