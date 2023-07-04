"""
    Author: Wang mingyuan
    Description: The model-arch for "RGB Depth Fusion GAN for Indoor Depth Completion"
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder.encoder_decoder import EncoderDecoder, conv_bn_relu
from .model_utils import AdaptiveInstanceNorm,AdaIN,IN
from .nlspn.nlspn_model import NLSPNRefineModule
from .segmentator.esa_net.esa_net_one_modality import ESANetOneModality

# encoder_decoder_module = ESANetOneModality(height=256,
#                                             width=320,
#                                             num_classes=3,
#                                             pretrained_on_imagenet=False,
#                                             encoder='resnet34',
#                                             encoder_block='BasicBlock',
#                                             channels_decoder=[512,256,128],
#                                             nr_decoder_blocks= [3,3,3],
#                                             encoder_decoder_fusion='add',
#                                             context_module='ppm',
#                                             weighting_in_encoder='SE-add',
#                                             upsampling='learned-3x3-zeropad',
#                                             pyramid_supervision=False)


class RDFGenerator(nn.Module):
    def __init__(self,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 pretrained_on_imagenet=True,
                 semantic_channels_in=3,
                 fuse_depth_in_rgb_decoder='WAdaIN',
                 bn=True,
                 rgb_skip_connection_type='concat',
                 depth_skip_connection_type='concat',
                 adain_weighting=False,
                 rgb_channels_encoder=[64, 64, 128, 256, 512, 512],  # output
                 depth_channels_encoder=[64, 64, 128, 256, 512, 512],
                 rgb_channels_decoder=[256, 128, 64, 64],  # output
                 depth_channels_decoder=[256, 128, 64, 64],
                 use_nlspn_refine=False,
                 nlspn_configs=None,
                 global_guidance_module=None   # The interface reserved for guidance module
                 ):
        super(RDFGenerator, self).__init__()

        self.use_nlspn_refine = use_nlspn_refine

        self.bn = bn
        self.rgb_skip_connection_type = rgb_skip_connection_type
        self.depth_skip_connection_type = depth_skip_connection_type

        self.global_guidance_module = global_guidance_module if global_guidance_module is not None else lambda x : x

        self.rgb_branch_en1 = conv_bn_relu(semantic_channels_in, rgb_channels_encoder[0], kernel=3,
                                           stride=1, padding=1, bn=False)
        self.rgb_branch_encoder_decoder = EncoderDecoder(encoder_type=encoder_rgb,
                                                         pretrained_on_imagenet=pretrained_on_imagenet,
                                                         skip_type=rgb_skip_connection_type,
                                                         encoder_channels=rgb_channels_encoder[1:],
                                                         decoder_channels=rgb_channels_decoder)

        self.rgb_pred_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.rgb_pred_dec0 = conv_bn_relu(64 + 64, 1, kernel=3, stride=1, padding=1, bn=False, relu=False)

        self.rgb_conf_dec1 = conv_bn_relu(64 + 64, 32, kernel=3, stride=1, padding=1)
        self.rgb_conf_dec0 = nn.Sequential(
            nn.Conv2d(32 + 64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.depth_branch_en1_rgb = conv_bn_relu(semantic_channels_in, 48, kernel=3,
                                                 stride=1, padding=1, bn=False)
        self.depth_branch_en1_depth = conv_bn_relu(1, 16, kernel=3, stride=1,
                                                   padding=1, bn=False)
        self.depth_branch_encoder_decoder = EncoderDecoder(encoder_type=encoder_depth,
                                                           pretrained_on_imagenet=pretrained_on_imagenet,
                                                           skip_type=depth_skip_connection_type,
                                                           encoder_channels=depth_channels_encoder[1:],
                                                           decoder_channels=depth_channels_decoder)
        # init depth branch
        self.id_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.id_dec0 = conv_bn_relu(64 + 64, 1, kernel=3, stride=1, padding=1, bn=False, relu=False)

        # guidance branch
        if self.use_nlspn_refine:
            num_neighbors = nlspn_configs['prop_kernel'] * nlspn_configs['prop_kernel'] - 1
            self.gd_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1,
                                        padding=1)
            self.gd_dec0 = conv_bn_relu(64 + 64, num_neighbors, kernel=3, stride=1, padding=1, bn=False, relu=False)

        # confidence branch
        self.cf_dec1 = conv_bn_relu(64 + 64, 32, kernel=3, stride=1, padding=1)
        self.cf_dec0 = nn.Sequential(
            nn.Conv2d(32 + 64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # prop layer
        if self.use_nlspn_refine:
            self.nlspn_refine_module = NLSPNRefineModule(**nlspn_configs)
        else:
            self.nlspn_refine_module = self.NLSPNIdentity()

        # rgb skip connection
        if rgb_skip_connection_type == 'add':
            self.rgb_skip_layer1 = self._build_skip_layer(rgb_channels_encoder[-2],
                                                          rgb_channels_decoder[0])
            self.rgb_skip_layer2 = self._build_skip_layer(rgb_channels_encoder[-3],
                                                          rgb_channels_decoder[1])
            self.rgb_skip_layer3 = self._build_skip_layer(rgb_channels_encoder[-4],
                                                          rgb_channels_decoder[2])
            self.rgb_skip_layer4 = self._build_skip_layer(rgb_channels_encoder[-5],
                                                          rgb_channels_decoder[3])
            self.rgb_skip_layer5 = self._build_skip_layer(rgb_channels_encoder[-6],
                                                          rgb_channels_decoder[4])
        else:
            self.rgb_skip_layer1 = nn.Identity()
            self.rgb_skip_layer2 = nn.Identity()
            self.rgb_skip_layer3 = nn.Identity()
            self.rgb_skip_layer4 = nn.Identity()
            self.rgb_skip_layer5 = nn.Identity()

        # depth skip connection
        if depth_skip_connection_type == 'add':
            self.depth_skip_layer1 = self._build_skip_layer(depth_channels_encoder[-2],
                                                            depth_channels_decoder[0])
            self.depth_skip_layer2 = self._build_skip_layer(depth_channels_encoder[-3],
                                                            depth_channels_decoder[1])
            self.depth_skip_layer3 = self._build_skip_layer(depth_channels_encoder[-4],
                                                            depth_channels_decoder[2])
            self.depth_skip_layer4 = self._build_skip_layer(depth_channels_encoder[-5],
                                                            depth_channels_decoder[3])
            self.depth_skip_layer5 = self._build_skip_layer(depth_channels_encoder[-6],
                                                            depth_channels_decoder[4])
        else:
            self.depth_skip_layer1 = nn.Identity()
            self.depth_skip_layer2 = nn.Identity()
            self.depth_skip_layer3 = nn.Identity()
            self.depth_skip_layer4 = nn.Identity()
            self.depth_skip_layer5 = nn.Identity()

        # rgb_channels_after_encoder = rgb_channels_encoder[-1]
        # depth_channels_after_encoder = depth_channels_encoder[-1]

        if fuse_depth_in_rgb_decoder == 'WAdaIN':
            self.fuse_layer1 = AdaptiveInstanceNorm(in_channel=rgb_channels_encoder[-1],
                                                    style_dim=rgb_channels_encoder[-1],
                                                    weighting=adain_weighting)
            self.fuse_layer2 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[0] + rgb_channels_encoder[-2] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[0],
                                                    style_dim=depth_channels_decoder[0] + depth_channels_encoder[-2] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[0],
                                                    weighting=adain_weighting)
            self.fuse_layer3 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[1] + rgb_channels_encoder[-3] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[1],
                                                    style_dim=depth_channels_decoder[1] + rgb_channels_encoder[-3] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[1],
                                                    weighting=adain_weighting)
            self.fuse_layer4 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[2] + rgb_channels_encoder[-4] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[2],
                                                    style_dim=depth_channels_decoder[2] + rgb_channels_encoder[-4] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[2],
                                                    weighting=adain_weighting)
            self.fuse_layer5 = AdaptiveInstanceNorm(in_channel=rgb_channels_decoder[3] + rgb_channels_encoder[-5] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[3],
                                                    style_dim=depth_channels_decoder[3] + rgb_channels_encoder[-5] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[3],
                                                    weighting=adain_weighting)
        elif fuse_depth_in_rgb_decoder == 'AdaIN':
            self.fuse_layer1 = AdaIN()
            self.fuse_layer2 = AdaIN()
            self.fuse_layer3 = AdaIN()
            self.fuse_layer4 = AdaIN()
            self.fuse_layer5 = AdaIN()
        elif fuse_depth_in_rgb_decoder == 'IN':
            self.fuse_layer1 = IN(in_channel=rgb_channels_encoder[-1],
                                                    style_dim=rgb_channels_encoder[-1])
            self.fuse_layer2 = IN(in_channel=rgb_channels_decoder[0] + rgb_channels_encoder[-2] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[0],
                                                    style_dim=depth_channels_decoder[0] + depth_channels_encoder[-2] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[0])
            self.fuse_layer3 = IN(in_channel=rgb_channels_decoder[1] + rgb_channels_encoder[-3] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[1],
                                                    style_dim=depth_channels_decoder[1] + rgb_channels_encoder[-3] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[1])
            self.fuse_layer4 = IN(in_channel=rgb_channels_decoder[2] + rgb_channels_encoder[-4] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[2],
                                                    style_dim=depth_channels_decoder[2] + rgb_channels_encoder[-4] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[2])
            self.fuse_layer5 = IN(in_channel=rgb_channels_decoder[3] + rgb_channels_encoder[-5] if \
                self.rgb_skip_connection_type == 'concat' else rgb_channels_decoder[3],
                                                    style_dim=depth_channels_decoder[3] + rgb_channels_encoder[-5] if \
                                                        self.rgb_skip_connection_type == 'concat' else
                                                    depth_channels_decoder[3])

        if self.rgb_skip_connection_type in ['concat', 'add']:
            self.rgb_skip_op = getattr(self, f'_{self.rgb_skip_connection_type}')
        else:
            # no skip connection
            self.rgb_skip_op = self.Identity()

        if self.depth_skip_connection_type in ['concat', 'add']:
            self.depth_skip_op = getattr(self, f'_{self.depth_skip_connection_type}')
        else:
            self.depth_skip_op = self.Identity()

        self.use_pretrained_global_guidance_module = False
        self.pretrained_on_imagenet = pretrained_on_imagenet

    def _build_skip_layer(self, ch_in, ch_out):
        layers_skip1 = list()
        if ch_in != ch_out:
            layers_skip1.append(conv_bn_relu(ch_in,
                                             ch_out,
                                             kernel=1,
                                             bn=self.bn,
                                             _in=not self.bn,
                                             relu=True))
        skip_layer = nn.Sequential(*layers_skip1)

        return skip_layer

    class Identity:
        def __call__(self, fd, fe=None):
            return fd

    class NLSPNIdentity():
        def __call__(self, pred_init, guide, confidence, origin_depth):
            return pred_init, confidence

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def _add(self, fd, fe):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = fd + fe

        return f

    def forward(self, rgb, depth, normal):
    #def forward(self, rgb, depth):
        origin_rgb, origin_depth= rgb, depth
        #global_guidance_map = self.global_guidance_module(rgb)    # (b, c, h, w)

        # encoder layer 1  [1/1]
        rgb_fe1 = self.rgb_branch_en1(normal)
        #rgb_fe1 = self.rgb_branch_en1(global_guidance_map)

        depth_fe1_rgb = self.depth_branch_en1_rgb(normal)
        #depth_fe1_rgb = self.depth_branch_en1_rgb(global_guidance_map)
        depth_fe1_depth = self.depth_branch_en1_depth(depth)
        depth_fe1 = torch.cat([depth_fe1_rgb, depth_fe1_depth], dim=1)

        # encoder layer 2  [1/1]
        rgb_fe2 = self.rgb_branch_encoder_decoder.forward_encoder_layer(rgb_fe1, layer_idx=2)
        depth_fe2 = self.depth_branch_encoder_decoder.forward_encoder_layer(depth_fe1, layer_idx=2)

        # encoder layer 3  [1/2]
        rgb_fe3 = self.rgb_branch_encoder_decoder.forward_encoder_layer(rgb_fe2, layer_idx=3)
        depth_fe3 = self.depth_branch_encoder_decoder.forward_encoder_layer(depth_fe2, layer_idx=3)

        # encoder layer 4  [1/4]
        rgb_fe4 = self.rgb_branch_encoder_decoder.forward_encoder_layer(rgb_fe3, layer_idx=4)
        depth_fe4 = self.depth_branch_encoder_decoder.forward_encoder_layer(depth_fe3, layer_idx=4)

        # encoder layer 5  [1/8]
        rgb_fe5 = self.rgb_branch_encoder_decoder.forward_encoder_layer(rgb_fe4, layer_idx=5)
        depth_fe5 = self.depth_branch_encoder_decoder.forward_encoder_layer(depth_fe4, layer_idx=5)

        # encoder layer 6  [1/16]
        rgb_fe6 = self.rgb_branch_encoder_decoder.forward_encoder_layer(rgb_fe5, layer_idx=6)
        depth_fe6 = self.depth_branch_encoder_decoder.forward_encoder_layer(depth_fe5, layer_idx=6)

        # decoder layer 5  [1/8]
        fuse = self.fuse_layer1(rgb_fe6, depth_fe6)
        rgb_fd5 = self.rgb_branch_encoder_decoder.forward_decoder_layer(fuse,
                                                                        # skip=self.rgb_skip5(rgb_fe5),
                                                                        # skip_type=self.skip_connection_type,
                                                                        layer_idx=5)
        rgb_fd5 = self.rgb_skip_op(rgb_fd5, self.rgb_skip_layer1(rgb_fe5))

        depth_fd5 = self.depth_branch_encoder_decoder.forward_decoder_layer(depth_fe6,
                                                                            # skip=self.depth_skip5(depth_fe5),
                                                                            # skip_type=self.skip_connection_type,
                                                                            layer_idx=5)
        depth_fd5 = self.depth_skip_op(depth_fd5, self.depth_skip_layer1(depth_fe5))

        # decoder layer 4  [1/4]
        fuse = self.fuse_layer2(rgb_fd5, depth_fd5)
        rgb_fd4 = self.rgb_branch_encoder_decoder.forward_decoder_layer(fuse,
                                                                        # skip=self.rgb_skip4(rgb_fe4),
                                                                        # skip_type=self.skip_connection_type,
                                                                        layer_idx=4)
        rgb_fd4 = self.rgb_skip_op(rgb_fd4, self.rgb_skip_layer2(rgb_fe4))

        depth_fd4 = self.depth_branch_encoder_decoder.forward_decoder_layer(depth_fd5,
                                                                            # skip=self.depth_skip4(depth_fe4),
                                                                            # skip_type=self.skip_connection_type,
                                                                            layer_idx=4)
        depth_fd4 = self.depth_skip_op(depth_fd4, self.depth_skip_layer2(depth_fe4))

        # decoder layer 3  [1/2]
        fuse = self.fuse_layer3(rgb_fd4, depth_fd4)
        rgb_fd3 = self.rgb_branch_encoder_decoder.forward_decoder_layer(fuse,
                                                                        # skip=self.rgb_skip3(rgb_fe3),
                                                                        # skip_type=self.skip_connection_type,
                                                                        layer_idx=3)
        rgb_fd3 = self.rgb_skip_op(rgb_fd3, self.rgb_skip_layer3(rgb_fe3))

        depth_fd3 = self.depth_branch_encoder_decoder.forward_decoder_layer(depth_fd4,
                                                                            # skip=self.depth_skip3(depth_fe3),
                                                                            # skip_type=self.skip_connection_type,
                                                                            layer_idx=3)
        depth_fd3 = self.depth_skip_op(depth_fd3, self.depth_skip_layer3(depth_fe3))

        # decoder layer 2  [1/1]
        fuse = self.fuse_layer4(rgb_fd3, depth_fd3)
        rgb_fd2 = self.rgb_branch_encoder_decoder.forward_decoder_layer(fuse,
                                                                        # skip=self.rgb_skip2(rgb_fe2),
                                                                        # skip_type=self.skip_connection_type,
                                                                        layer_idx=2)
        rgb_fd2 = self.rgb_skip_op(rgb_fd2, self.rgb_skip_layer4(rgb_fe2))

        depth_fd2 = self.depth_branch_encoder_decoder.forward_decoder_layer(depth_fd3,
                                                                            # skip=self.depth_skip2(depth_fe2),
                                                                            # skip_type=self.skip_connection_type,
                                                                            layer_idx=2)
        depth_fd2 = self.depth_skip_op(depth_fd2, self.depth_skip_layer4(depth_fe2))

        # for rgb branch, use gan to generate depth map
        # fuse = self.fuse_layer5(rgb_fd2, depth_fd2)
        skipped_rgb_fe1 = self.rgb_skip_layer5(rgb_fe1)

        rgb_pred_fd1 = self.rgb_pred_dec1(rgb_fd2)
        depth_map_1 = self.rgb_pred_dec0(self.rgb_skip_op(rgb_pred_fd1, skipped_rgb_fe1))
        depth_map_1 = torch.tanh(depth_map_1)

        rgb_conf_fd1 = self.rgb_conf_dec1(rgb_fd2)
        confidence_map_1 = self.rgb_conf_dec0(self.rgb_skip_op(rgb_conf_fd1, skipped_rgb_fe1))

        # for depth branch, use nlpsn-refine to generate depth map
        # init depth decoding
        skipped_depth_fe1 = self.depth_skip_layer5(depth_fe1)

        id_fd1 = self.id_dec1(depth_fd2)
        pred_init = self.id_dec0(self.depth_skip_op(id_fd1, skipped_depth_fe1))
        pred_init = torch.tanh(pred_init)

        # guidance decoding
        if self.use_nlspn_refine:
            gd_fd1 = self.gd_dec1(depth_fd2)
            guide = self.gd_dec0(self.depth_skip_op(gd_fd1, skipped_depth_fe1))
        else:
            guide = None

        # confidence decoding
        cf_fd1 = self.cf_dec1(depth_fd2)
        confidence = self.cf_dec0(self.depth_skip_op(cf_fd1, skipped_depth_fe1))

        depth_map_2, confidence_map_2 = self.nlspn_refine_module(pred_init, guide, confidence, origin_depth)
        depth_map_2 = torch.clamp(depth_map_2, min=-1, max=1)

        confidence_map = torch.cat([confidence_map_1, confidence_map_2], dim=1)
        confidence_score = F.softmax(confidence_map, 1)
        final_depth_map = torch.cat([depth_map_1, depth_map_2], dim=1)
        final_depth_map = torch.sum(final_depth_map * confidence_score, dim=1, keepdim=True)

        # return depth_map_1, confidence_map_1, depth_map_2, confidence_map_2, final_depth_map
        ret = dict(depth_map_1=depth_map_1,
                   confidence_map_1=confidence_map_1,
                   depth_map_2=depth_map_2,
                   confidence_map_2=confidence_map_2,
                   pred_depth=final_depth_map)
        return ret
