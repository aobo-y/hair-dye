import torch
import torch.nn as nn
import torch.nn.functional as F

################################
## Encoding part unit ##
################################


class _GreenBlock(nn.Module):
    ''' The Green Block, including a depthwise 3x3 conv,
        BatchNorm, Relu, 1x1 conv, BatchNorm, Relu '''

    def __init__(self, in_channels, out_channels, stride, kernel_size=3):
        super(_GreenBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        output = self.layer(x)
        return output


class _BlueBlock(nn.Module):
    ''' The Blue Block. just a 3x3 convolution. '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(_BlueBlock, self).__init__()

        self.layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        output = self.layer(x)
        return output


class _EncodeBlockBlueGreen(nn.Module):
    ''' The encode block, including a blue block and a green block '''

    def __init__(self, in_channels, out_channels):
        super(_EncodeBlockBlueGreen, self).__init__()

        self.layer = nn.Sequential(
            _BlueBlock(in_channels, 32),
            _GreenBlock(32, out_channels, 1)
        )

    def forward(self, x):
        output = self.layer(x)
        return output


class _EncodeBlockGreenGreen(nn.Module):
    ''' The encode block, including a green block and a green block '''

    def __init__(self, in_channels, out_channels):
        super(_EncodeBlockGreenGreen, self).__init__()

        self.layer = nn.Sequential(
            _GreenBlock(in_channels, 2 * in_channels, 2),
            _GreenBlock(2 * in_channels, out_channels, 1)
        )

    def forward(self, x):
        output = self.layer(x)
        return output


class _EncodeBlock6Green(nn.Module):
    ''' The encode block, including 6 green blocks '''

    def __init__(self, in_channels, out_channels):
        super(_EncodeBlock6Green, self).__init__()

        self.layer = nn.Sequential(
            _GreenBlock(in_channels,     2 * in_channels, 2),
            _GreenBlock(2 * in_channels, 2 * in_channels, 1),
            _GreenBlock(2 * in_channels, 2 * in_channels, 1),
            _GreenBlock(2 * in_channels, 2 * in_channels, 1),
            _GreenBlock(2 * in_channels, 2 * in_channels, 1),
            _GreenBlock(2 * in_channels, out_channels, 1)
        )

    def forward(self, x):
        output = self.layer(x)
        return output

################################
## Decoding part unit ##
################################


class _SkipConnection(nn.Module):
    ''' The skip connection. just a 1x1 convolution. '''

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(_SkipConnection, self).__init__()

        self.layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        output = self.layer(x)
        return output


class _UpsampleLayer(nn.Module):
    ''' upsampling layer '''

    def __init__(self):
        super(_UpsampleLayer, self).__init__()

    def forward(self, x):
        output = F.interpolate(x, scale_factor=2)
        return output


class _YellowBlock(nn.Module):
    ''' Yellow block, add the upsampling layer and Conv 1x1 layer '''

    def __init__(self, in_channels, out_channels):
        super(_YellowBlock, self).__init__()

        self.upsample    = _UpsampleLayer()
        self.skipConnect = _SkipConnection(in_channels, out_channels)

    def forward(self, upsample_input, skipConnect_input):
        upsample_input    = self.upsample(upsample_input)
        skipConnect_input = self.skipConnect(skipConnect_input)
        output            = torch.add(upsample_input, skipConnect_input)
        return output


class _OrangeBlock(nn.Module):
    ''' The Orange Block, including a depthwise 3x3 conv, 1x1 conv, Relu '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(_OrangeBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=stride),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class _RedBlock(nn.Module):
    ''' The Red Block, including a 1x1 conv, softmax '''

    def __init__(self, in_channels, out_channels=1, kernel_size=1, stride=1):
        super(_RedBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class _DecodeBlockYellowOrange(nn.Module):
    ''' The decode block, include a yellow block and orange block '''

    def __init__(self, in_channels_skip, out_channels_yellow, out_channels):
        super(_DecodeBlockYellowOrange, self).__init__()

        self.yellow_layer = _YellowBlock(in_channels_skip, out_channels_yellow)
        self.orange_layer = _OrangeBlock(out_channels_yellow, out_channels)

    def forward(self, upsample_input, skipConnect_input):
        yellow_out = self.yellow_layer(upsample_input, skipConnect_input)
        output     = self.orange_layer(yellow_out)
        return output


class _DecodeBlockYellowOrangeRed(nn.Module):
    ''' The decode block, include a upsamling block, a orange block and a red block '''

    def __init__(self, in_channels, out_channels=1):
        super(_DecodeBlockYellowOrangeRed, self).__init__()

        self.upsample_layer = _UpsampleLayer()
        self.orange_layer   = _OrangeBlock(in_channels, in_channels)
        self.red_layer      = _RedBlock(in_channels, out_channels)

    def forward(self, x):
        x      = self.upsample_layer(x)
        x      = self.orange_layer(x)
        output = self.red_layer(x)
        return output
