import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import UpSampleBN, UpSampleGN, norm_normalize, sample_points

class Decoder(nn.Module):
    def __init__(self, num_classes=4):
        super(Decoder, self).__init__()
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
        self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out
    

