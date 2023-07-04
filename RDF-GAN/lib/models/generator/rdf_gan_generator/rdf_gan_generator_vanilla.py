import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbone.resnet import (ResNet18, ResNet34, ResNet50, ResNet)
from .model_utils import DCVGANDecoderModule, ConvNormAct
from .model_utils import AdaptiveInstanceNorm


class DCVGANGenerator(nn.Module):
    def __init__(self,
                 global_guidance_module: nn.Module,
                 global_guidance_module_out_channels_0=1,
                 global_guidance_module_out_channels_1=1,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 rgb_channels_decoder=None,
                 depth_channels_decoder=None,
                 nr_decoder_blocks= None,
                 pretrained_on_imagenet=True,
                 pretrained_dir='./pretrained_model/resnet_on_imagenet',
                 fuse_depth_in_rgb_encoder=None,
                 fuse_depth_in_rgb_decoder='AdaIN',
                 encoder_decoder_fusion='add',
                 activation='relu',
                 norm_layer_type=None,
                 upsampling_mode='bilinear',
                 adain_weighting=False,
                 separate_global_guidance_module=False,
                 use_pretrained_global_guidance_module=False):
        super(DCVGANGenerator, self).__init__()

        self.fuse_depth_in_rgb_decoder = fuse_depth_in_rgb_decoder

        self.global_guidance_module = global_guidance_module

        if rgb_channels_decoder is None:
            rgb_channels_decoder = [128, 128, 128, 128, 128]
        if depth_channels_decoder is None:
            depth_channels_decoder = [128, 128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1, 0, 0]

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2,
                                           inplace=True)
        else:
            raise NotImplementedError(f'Only ReLU is supported currently')

        # the rgb_branch contains two encoder-decoder arch
        self.rgb_branch = None

        # rgb encoder
        rgb_encoder_channels_in = global_guidance_module_out_channels_0
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(block=encoder_block,
                                        pretrained_on_imagenet=pretrained_on_imagenet,
                                        pretrained_dir=pretrained_dir,
                                        activation=self.activation,
                                        input_channels=rgb_encoder_channels_in)
            rgb_channels_after_encoder = 512
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(block=encoder_block,
                                        pretrained_on_imagenet=pretrained_on_imagenet,
                                        pretrained_dir=pretrained_dir,
                                        activation=self.activation,
                                        input_channels=rgb_encoder_channels_in)
            rgb_channels_after_encoder = 512
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(pretrained_on_imagenet=pretrained_on_imagenet,
                                        pretrained_dir=pretrained_dir,
                                        activation=self.activation,
                                        input_channels=rgb_encoder_channels_in)
            rgb_channels_after_encoder = 2048
        else:
            raise NotImplementedError(
                'Only ResNets are supported, Got {}'.format(encoder_rgb))

        # depth encoder
        depth_encoder_channels_in = global_guidance_module_out_channels_1 + 1
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(block=encoder_block,
                                          pretrained_on_imagenet=pretrained_on_imagenet,
                                          pretrained_dir=pretrained_dir,
                                          activation=self.activation,
                                          input_channels=depth_encoder_channels_in)
            depth_channels_after_encoder = 512
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(block=encoder_block,
                                          pretrained_on_imagenet=pretrained_on_imagenet,
                                          pretrained_dir=pretrained_dir,
                                          activation=self.activation,
                                          input_channels=depth_encoder_channels_in)
            depth_channels_after_encoder = 512
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(pretrained_on_imagenet=pretrained_on_imagenet,
                                          pretrained_dir=pretrained_dir,
                                          activation=self.activation,
                                          input_channels=depth_encoder_channels_in)
            depth_channels_after_encoder = 2048
        else:
            raise NotImplementedError(
                'Only ResNets are supported, Got {}'.format(encoder_depth))

        assert fuse_depth_in_rgb_encoder is None, f'No fusion in encoder stage so far'

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != rgb_channels_decoder[2]:
                # adjust the dimension for skip connection
                layers_skip1.append(ConvNormAct(self.encoder_rgb.down_4_channels_out,
                                                rgb_channels_decoder[2],
                                                kernel_size=1,
                                                norm_layer_type=norm_layer_type,
                                                activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != rgb_channels_decoder[1]:
                layers_skip2.append(ConvNormAct(self.encoder_rgb.down_8_channels_out,
                                                rgb_channels_decoder[1],
                                                kernel_size=1,
                                                norm_layer_type=norm_layer_type,
                                                activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != rgb_channels_decoder[0]:
                layers_skip3.append(ConvNormAct(self.encoder_rgb.down_16_channels_out,
                                                rgb_channels_decoder[0],
                                                kernel_size=1,
                                                norm_layer_type=norm_layer_type,
                                                activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)
        elif encoder_decoder_fusion == 'None':
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # Whether the dimension match or not, the AdaIN module transforms the dimension into the rgb
        # branch input dimension

        if fuse_depth_in_rgb_decoder == 'AdaIN':

            # the depth is the style
            self.fuse_layer1 = AdaptiveInstanceNorm(in_channel=rgb_channels_after_encoder,
                                                    style_dim=depth_channels_after_encoder,
                                                    weighting=adain_weighting)
            self.fuse_layer2 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[0],
                                                    style_dim=depth_channels_decoder[0],
                                                    weighting=adain_weighting)
            self.fuse_layer3 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[1],
                                                    style_dim=depth_channels_decoder[1],
                                                    weighting=adain_weighting)
            self.fuse_layer4 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[2],
                                                    style_dim=depth_channels_decoder[2],
                                                    weighting=adain_weighting)
            self.fuse_layer5 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[3],
                                                    style_dim=depth_channels_decoder[3],
                                                    weighting=adain_weighting)
            self.fuse_layer6 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[4],
                                                    style_dim=depth_channels_decoder[4],
                                                    weighting=adain_weighting)
        else:
            raise NotImplementedError(f'Only support AdaIN for fuison depth and rgb,'
                                      f'but got {fuse_depth_in_rgb_decoder}')



        # rgb decoder
        self.decoder_rgb_1 = DCVGANDecoderModule(channels_in=rgb_channels_after_encoder,
                                                 channels_out=rgb_channels_decoder[0],
                                                 activation=self.activation,
                                                 nr_decoder_blocks=nr_decoder_blocks[0],
                                                 norm_layer_type=norm_layer_type,
                                                 encoder_decoder_fusion=encoder_decoder_fusion,
                                                 upsampling_mode=upsampling_mode)
        self.decoder_rgb_2 = DCVGANDecoderModule(channels_in=rgb_channels_decoder[0],
                                                 channels_out=rgb_channels_decoder[1],
                                                 activation=self.activation,
                                                 nr_decoder_blocks=nr_decoder_blocks[1],
                                                 norm_layer_type=norm_layer_type,
                                                 encoder_decoder_fusion=encoder_decoder_fusion,
                                                 upsampling_mode=upsampling_mode)
        self.decoder_rgb_3 = DCVGANDecoderModule(channels_in=rgb_channels_decoder[1],
                                                 channels_out=rgb_channels_decoder[2],
                                                 activation=self.activation,
                                                 nr_decoder_blocks=nr_decoder_blocks[2],
                                                 norm_layer_type=norm_layer_type,
                                                 encoder_decoder_fusion=encoder_decoder_fusion,
                                                 upsampling_mode=upsampling_mode)
        self.decoder_rgb_4 = DCVGANDecoderModule(channels_in=rgb_channels_decoder[2],
                                                 channels_out=rgb_channels_decoder[3],
                                                 activation=self.activation,
                                                 nr_decoder_blocks=nr_decoder_blocks[3],
                                                 norm_layer_type=norm_layer_type,
                                                 encoder_decoder_fusion=None,
                                                 upsampling_mode=upsampling_mode)
        self.decoder_rgb_5 = DCVGANDecoderModule(channels_in=rgb_channels_decoder[3],
                                                 channels_out=rgb_channels_decoder[4],
                                                 activation=self.activation,
                                                 nr_decoder_blocks=nr_decoder_blocks[4],
                                                 norm_layer_type=norm_layer_type,
                                                 encoder_decoder_fusion=None,
                                                 upsampling_mode=upsampling_mode)

        # depth decoder
        self.decoder_depth_1 = DCVGANDecoderModule(channels_in=depth_channels_after_encoder,
                                                   channels_out=depth_channels_decoder[0],
                                                   activation=self.activation,
                                                   nr_decoder_blocks=nr_decoder_blocks[0],
                                                   norm_layer_type=norm_layer_type,
                                                   encoder_decoder_fusion=None,   # no encoder decoder fusion in depth branch
                                                   upsampling_mode=upsampling_mode)
        self.decoder_depth_2 = DCVGANDecoderModule(channels_in=depth_channels_decoder[0],
                                                   channels_out=depth_channels_decoder[1],
                                                   activation=self.activation,
                                                   nr_decoder_blocks=nr_decoder_blocks[1],
                                                   norm_layer_type=norm_layer_type,
                                                   encoder_decoder_fusion=None,
                                                   upsampling_mode=upsampling_mode)
        self.decoder_depth_3 = DCVGANDecoderModule(channels_in=depth_channels_decoder[1],
                                                   channels_out=depth_channels_decoder[2],
                                                   activation=self.activation,
                                                   nr_decoder_blocks=nr_decoder_blocks[2],
                                                   norm_layer_type=norm_layer_type,
                                                   encoder_decoder_fusion=None,
                                                   upsampling_mode=upsampling_mode)
        self.decoder_depth_4 = DCVGANDecoderModule(channels_in=depth_channels_decoder[2],
                                                   channels_out=depth_channels_decoder[3],
                                                   activation=self.activation,
                                                   nr_decoder_blocks=nr_decoder_blocks[3],
                                                    norm_layer_type=norm_layer_type,
                                                   encoder_decoder_fusion=None,
                                                   upsampling_mode=upsampling_mode)
        self.decoder_depth_5 = DCVGANDecoderModule(channels_in=depth_channels_decoder[3],
                                                   channels_out=depth_channels_decoder[4],
                                                   activation=self.activation,
                                                   nr_decoder_blocks=nr_decoder_blocks[4],
                                                   norm_layer_type=norm_layer_type,
                                                   encoder_decoder_fusion=None,
                                                   upsampling_mode=upsampling_mode)

        self.rgb_conv_0 = nn.Conv2d(rgb_channels_decoder[-1], 1, kernel_size=3, padding=1)
        self.rgb_conv_1 = nn.Conv2d(rgb_channels_decoder[-1], 1, kernel_size=3, padding=1)

        self.depth_conv_0 = nn.Conv2d(depth_channels_decoder[-1], 1, kernel_size=3, padding=1)
        self.depth_conv_1 = nn.Conv2d(depth_channels_decoder[-1], 1, kernel_size=3, padding=1)

        self.separate_global_guidance_module = separate_global_guidance_module
        self.pretrained_on_imagenet = pretrained_on_imagenet
        self.use_pretrained_global_guidance_module = use_pretrained_global_guidance_module

        self.init_weight()

    def init_weight(self):
        module_list = []
        from lib.models.segmentator.esa_net.esa_net_one_modality import ESANetOneModality
        # filter out modules that do not need to be initialized
        for name, c in self.named_children():
            if name == 'global_guidance_module':
                # ESAOneModality + nn.Conv2d
                for m in c.children():
                    if self.use_pretrained_global_guidance_module and isinstance(m, ESANetOneModality):
                        continue
                    for mm in m.modules():
                        module_list.append(mm)
            else:
                if self.pretrained_on_imagenet and isinstance(c, ResNet):
                    continue
                if isinstance(c, AdaptiveInstanceNorm):
                    continue
                for m in c.modules():
                    module_list.append(m)

        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                try:
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_in'
                                            )
                except AttributeError:
                    import pdb
                    pdb.set_trace()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print(f'Apply kaiming init on {self.__class__.__name__}')


    def forward(self, rgb, depth, idx=0):
        # global guidance model
        if not self.separate_global_guidance_module:
            rgb = self.global_guidance_module(rgb)
            fuse = torch.cat([depth, rgb], dim=1)
        else:
            rgb = self.global_guidance_module[0](rgb)
            guid_info = self.global_guidance_module[1](rgb)
            fuse = torch.cat([depth, guid_info], dim=1)



        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(fuse)

        rgb = F.max_pool2d(rgb, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # encoder block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        skip1 = self.skip_layer1(rgb)

        # encoder block 2
        rgb = self.encoder_rgb.forward_layer2(rgb)
        depth = self.encoder_depth.forward_layer2(depth)
        skip2 = self.skip_layer2(rgb)

        # encoder block 3
        rgb = self.encoder_rgb.forward_layer3(rgb)
        depth = self.encoder_depth.forward_layer3(depth)
        skip3 = self.skip_layer3(rgb)

        # encoder block 4
        rgb = self.encoder_rgb.forward_layer4(rgb)
        depth = self.encoder_depth.forward_layer4(depth)

        # decoder block 1
        fuse = self.fuse_layer1(rgb, depth)
        rgb = self.decoder_rgb_1(fuse, skip3)
        depth = self.decoder_depth_1(depth, up_size=tuple(skip3.shape[-2:]))

        # decoder block 2
        fuse = self.fuse_layer2(rgb, depth)
        rgb = self.decoder_rgb_2(fuse, skip2)
        depth = self.decoder_depth_2(depth, up_size=tuple(skip2.shape[-2:]))

        # decoder block 3
        fuse = self.fuse_layer3(rgb, depth)
        rgb = self.decoder_rgb_3(fuse, skip1)
        depth = self.decoder_depth_3(depth, up_size=tuple(skip1.shape[-2:]))

        # there is no skip connection in decoder block 4 and 5
        # decoder block 4
        fuse = self.fuse_layer4(rgb, depth)
        rgb = self.decoder_rgb_4(fuse)
        depth = self.decoder_depth_4(depth)

        # decoder block 5
        fuse = self.fuse_layer5(rgb, depth)
        rgb = self.decoder_rgb_5(fuse)
        depth = self.decoder_depth_5(depth)

        # before generate depth map and confidence map, pass a AdaIN module (rgb branch)
        rgb = self.fuse_layer6(rgb, depth)

        depth_map_1 = self.rgb_conv_0(rgb)
        confidence_map_1 = self.rgb_conv_1(rgb)

        depth_map_2 = self.depth_conv_0(depth)
        confidence_map_2 = self.depth_conv_1(depth)

        depth_map_1 = torch.tanh(depth_map_1)
        depth_map_2 = torch.tanh(depth_map_2)

        confidence_map = torch.cat([confidence_map_1, confidence_map_2], dim=1)
        confidence_score = F.softmax(confidence_map, 1)
        final_depth_map = torch.cat([depth_map_1, depth_map_2], dim=1)
        final_depth_map = torch.sum(final_depth_map * confidence_score, dim=1, keepdim=True)

        return depth_map_1, confidence_map_1, depth_map_2, confidence_map_2, final_depth_map
