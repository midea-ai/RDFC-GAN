import torch
from .common import *


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder_type='resnet34',
                 skip_type='concat',
                 encoder_channels=[64, 128, 256, 512, 512],
                 decoder_channels=[256, 128, 64, 64],
                 pretrained_on_imagenet=False):
        super(EncoderDecoder, self).__init__()

        # encoder
        # self.en1_rgb = conv_bn_relu(rgb_channels_in, 48, kernel=3, stride=1, padding=1,
        #                               bn=False)
        # self.en1_depth = conv_bn_relu(depth_channels_in, 16, kernel=3, stride=1, padding=1,
        #                                 bn=False)

        if encoder_type == 'resnet18':
            net = get_resnet18(pretrained_on_imagenet)
        elif encoder_type == 'resnet34':
            net = get_resnet34(pretrained_on_imagenet)
        else:
            raise NotImplementedError

        # encoder_channels = [64, 128, 256, 512, 512]    # out channels
        # decoder_channels = [256, 128, 64, 64]         # out_channels

        decoder_channels = [(encoder_channels[-1], decoder_channels[0]),
                            (decoder_channels[0] + encoder_channels[-2] if skip_type == 'concat' else decoder_channels[0],
                             decoder_channels[1]),
                            (decoder_channels[1] + encoder_channels[-3] if skip_type == 'concat' else decoder_channels[1],
                             decoder_channels[2]),
                            (decoder_channels[2] + encoder_channels[-4] if skip_type == 'concat' else decoder_channels[2],
                             decoder_channels[3]),
                            ]

        # 1/1
        self.en2 = net.layer1
        # 1/2
        self.en3 = net.layer2
        # 1/4
        self.en4 = net.layer3
        # 1/8
        self.en5 = net.layer4

        del net

        # 1/16
        self.en6 = conv_bn_relu(encoder_channels[-2], encoder_channels[-1], kernel=3, stride=2, padding=1)

        # decoder
        # 1/8
        self.de5 = convt_bn_relu(decoder_channels[0][0], decoder_channels[0][1], kernel=3, stride=2, padding=1, output_padding=1)
        # 1/4
        self.de4 = convt_bn_relu(decoder_channels[1][0], decoder_channels[1][1], kernel=3, stride=2, padding=1, output_padding=1)
        # 1/2
        self.de3 = convt_bn_relu(decoder_channels[2][0], decoder_channels[2][1], kernel=3, stride=2, padding=1, output_padding=1)
        # 1/1
        self.de2 = convt_bn_relu(decoder_channels[3][0], decoder_channels[3][1], kernel=3, stride=2, padding=1, output_padding=1)


        # # init depth branch
        # self.id_de1 = conv_bn_relu(64+64, 64, kernel=3, stride=1, padding=1)
        # self.id_de0 = conv_bn_relu(64+64, 1, kernel=3, stride=1, padding=1, bn=False, relu=False)

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

    def __add(self, fd, fe):
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

    def forward_encoder_layer(self, x, layer_idx=2):
        assert layer_idx > 1

        out = getattr(self, f'en{layer_idx}')(x)
        return out

    def forward_decoder_layer(self, x, skip=None, skip_type='concat', layer_idx=2):
        assert layer_idx > 1

        out = getattr(self, f'de{layer_idx}')(x)

        # if skip is not None:
        #     if skip_type == 'concat':
        #         out = self._concat(out, skip)
        #     elif skip_type == 'add':
        #         out = self.__add(out, skip)
        #     else:
        #         raise NotImplementedError
        return out


    def forward(self, rgb, depth):

        fe1_rgb = self.en1_rgb(rgb)
        fe1_depth = self.en1_depth(depth)

        fe1 = torch.cat([fe1_rgb, fe1_depth], dim=1)

        # encoding
        fe2 = self.en2(fe1)
        fe3 = self.en3(fe2)
        fe4 = self.en4(fe3)
        fe5 = self.en5(fe4)
        fe6 = self.en6(fe5)

        # decoding
        fd5 = self.de5(fe6)
        fd4 = self.de4(self._concat(fd5, fe5))
        fd3 = self.de3(self._concat(fd4, fe4))
        fd2 = self.de2(self._concat(fd3, fe3))

        # # init depth decoding
        # id_fd1 = self.id_de1(self._concat(fd2, fe2))
        # pred_init_depth = self.id_de0(self._concat(id_fd1, fe1))



