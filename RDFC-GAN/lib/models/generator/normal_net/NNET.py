import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules.encoder import Encoder
from .submodules.decoder import Decoder


class NNET(nn.Module):
    def __init__(self,out_height,out_width):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.out_height = out_height
        self.out_width = out_width

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()

    def forward(self, img):
        out = self.decoder(self.encoder(img))
        out = F.interpolate(out, size=[self.out_height, self.out_width], mode='bilinear', align_corners=True)
        return out