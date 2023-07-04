import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.models.backbone.resnet import (ResNet18, ResNet34, ResNet50, ResNet)
from .model_utils import SqueezeAndExcitation, ConvBNAct, ConvBN, get_context_module
from .decoder import Decoder
from ..base import BaseSegmentator


class ESANetOneModality(BaseSegmentator):
    """`ESANet <https://arxiv.org/pdf/2011.06961.pdf>`
        Single modal input
    """

    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,
                 pretrained_on_imagenet=True,
                 pretrained_dir='./pretrained_model/resnet_on_imagenet',
                 activation='relu',
                 input_channels=3,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,
                 weighting_in_encoder='None',
                 upsampling='bilinear',
                 pyramid_supervision=True):
        super(ESANetOneModality, self).__init__()

        self.pretrained_on_imagenet = pretrained_on_imagenet
        self.num_classes = num_classes

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.weighting_in_encoder = weighting_in_encoder

        if activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError(f'Only ReLU is supported currently')

        # encoder
        if encoder == 'resnet18':
            self.encoder = ResNet18(pretrained_on_imagenet=pretrained_on_imagenet,
                                    pretrained_dir=pretrained_dir,
                                    block=encoder_block,
                                    activation=self.act,
                                    input_channels=input_channels)
        elif encoder == 'resnet34':
            self.encoder = ResNet34(pretrained_on_imagenet=pretrained_on_imagenet,
                                    pretrained_dir=pretrained_dir,
                                    block=encoder_block,
                                    activation=self.act,
                                    input_channels=input_channels)
        elif encoder == 'resnet50':
            self.encoder = ResNet50(pretrained_on_imagenet=pretrained_on_imagenet,
                                    pretrained_dir=pretrained_dir,
                                    activation=self.act,
                                    input_channels=input_channels)
        else:
            raise NotImplementedError(f'Only resnet18, resnet34, resnet50 encoders are'
                                      f'supported so far, but got {encoder}')

        # TODO: fetch this property from encoder internal, a trivial modification on ResNet Module is required
        self.channels_decoder_in = self.encoder.down_32_channels_out

        if weighting_in_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExcitation(64, activation=self.act)
            self.se_layer1 = SqueezeAndExcitation(self.encoder.down_4_channels_out, activation=self.act)
            self.se_layer2 = SqueezeAndExcitation(self.encoder.down_8_channels_out, activation=self.act)
            self.se_layer3 = SqueezeAndExcitation(self.encoder.down_16_channels_out, activation=self.act)
            self.se_layer4 = SqueezeAndExcitation(self.encoder.down_32_channels_out, activation=self.act)
        else:
            self.se_layer0 = nn.Identity()
            self.se_layer1 = nn.Identity()
            self.se_layer2 = nn.Identity()
            self.se_layer3 = nn.Identity()
            self.se_layer4 = nn.Identity()

        if encoder_decoder_fusion == 'add':
            layers_skip1 = []
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(self.encoder.down_4_channels_out,
                                              channels_decoder[2],
                                              kernel_size=1,
                                              activation=self.act))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = []
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(self.encoder.down_8_channels_out,
                                              channels_decoder[1],
                                              kernel_size=1,
                                              activation=self.act))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = []
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(self.encoder.down_16_channels_out,
                                              channels_decoder[0],
                                              kernel_size=1,
                                              activation=self.act))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling

        self.context_module, channels_after_context_module = get_context_module(context_module,
                                                                                self.channels_decoder_in,
                                                                                channels_decoder[0],
                                                                                input_size=(height // 32, width // 32),
                                                                                activation=self.act,
                                                                                upsampling_mode=upsampling_context_module)
        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.act,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes,
            pyramid_supervision=pyramid_supervision)

        self.init_weights()


    def forward_net(self, image):
        # import pdb
        # pdb.set_trace()
        out = self.encoder.forward_first_conv(image)                     # down_2
        out = self.se_layer0(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)     # down_4

        # block 1
        out = self.encoder.forward_layer1(out)
        out = self.se_layer1(out)
        skip1 = self.skip_layer1(out)

        # block 2
        out = self.encoder.forward_layer2(out)
        out = self.se_layer2(out)
        skip2 = self.skip_layer2(out)

        # block 3
        out = self.encoder.forward_layer3(out)
        out = self.se_layer3(out)
        skip3 = self.skip_layer3(out)

        # block 4
        out = self.encoder.forward_layer4(out)
        out = self.se_layer4(out)

        out = self.context_module(out)

        outs = [out, skip3, skip2, skip1]

        return self.decoder(enc_outs=outs)

    def init_weights(self):
        module_list = []
        for c in self.children():
            if self.pretrained_on_imagenet and isinstance(c, ResNet):
                continue
            for m in c.modules():
                module_list.append(m)

        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == self.num_classes or \
                    isinstance(module_list[i+1], nn.Sigmoid) or \
                    m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init')


def main():
    # import pdb
    # pdb.set_trace()
    model = ESANetOneModality(encoder='resnet50')
    print(model)

    model.eval()

    image = torch.randn((1, 3, 480, 640))

    with torch.no_grad():
        output = model(image)
    print(output.shape)


if __name__ == '__main__':
    main()
