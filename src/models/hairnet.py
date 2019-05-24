import torch
import torch.nn as nn
from torch.nn import functional as F
from .unit import _EncodeBlockBlueGreen, _EncodeBlockGreenGreen, _EncodeBlock6Green
from .unit import _DecodeBlockYellowOrange, _DecodeBlockYellowOrangeRed

class MobileHairNet(nn.Module):
    def __init__(self, im_size=224):
        super(MobileHairNet, self).__init__()

        # Encoder
        self.encode_block1 = _EncodeBlockBlueGreen(3, 64) # in_channels and out_channels
        self.encode_block2 = _EncodeBlockGreenGreen(64, 128)
        self.encode_block3 = _EncodeBlockGreenGreen(128, 256)
        self.encode_block4 = _EncodeBlock6Green(256, 512)
        self.encode_block5 = _EncodeBlockGreenGreen(512, 1024)

        # Decoder
        # in_skip_channels, out_channels_yellow, out_channels
        self.decode_block1 = _DecodeBlockYellowOrange(512, 1024, 64)
        self.decode_block2 = _DecodeBlockYellowOrange(256, 64, 64)
        self.decode_block3 = _DecodeBlockYellowOrange(128, 64, 64)
        self.decode_block4 = _DecodeBlockYellowOrange(64, 64, 64)
        self.decode_block5 = _DecodeBlockYellowOrangeRed(64)

    def forward(self, x):
        encode_block1 = self.encode_block1(x)
        encode_block2 = self.encode_block2(encode_block1)
        encode_block3 = self.encode_block3(encode_block2)
        encode_block4 = self.encode_block4(encode_block3)
        encode_block5 = self.encode_block5(encode_block4)

        decode_block1 = self.decode_block1(encode_block5, encode_block4)
        decode_block2 = self.decode_block2(decode_block1, encode_block3)
        decode_block3 = self.decode_block3(decode_block2, encode_block2)
        decode_block4 = self.decode_block4(decode_block3, encode_block1)
        output        = self.decode_block5(decode_block4)

        return output
