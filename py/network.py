import torch
import torch.nn as nn


class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # two-convs
        self.TCB1 = TwoConvBlock(1, 32)
        self.TCB2 = TwoConvBlock(32, 64)
        self.TCB3 = TwoConvBlock(64, 128)

        self.TCB4 = TwoConvBlock(128, 256)

        self.TCB5 = TwoConvBlock(256, 128)
        self.TCB6 = TwoConvBlock(128, 64)
        self.TCB7 = TwoConvBlock(64, 32)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        # up-convs
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 

        self.conv1 = nn.Conv2d(32, 1, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x) # 20x20x32

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x) # 10x10x64

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x) # 5x5x128

        # middle
        x = self.TCB4(x) # 5x5x256

        # decoder
        x = self.UC1(x) # 10x10x128
        x = torch.cat([x3, x], dim = 1) # 10x10x256
        x = self.TCB5(x) # 10x10x128

        x = self.UC2(x) # 20x20x64
        x = torch.cat([x2, x], dim = 1) # 20x20x128
        x = self.TCB6(x) # 20x20x64

        x = self.UC3(x) # 40x40x32
        x = torch.cat([x1, x], dim = 1) # 40x40x64
        x = self.TCB7(x) # 40x40x32

        x = self.conv1(x) # 40x40x1
        x = self.sigmoid(x)

        return x
