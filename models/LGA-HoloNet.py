#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Project    : neural-holography
# @FileName   : LGA-HoloNet.py
# @Author     : Spring
# @Time       : 2023/5/22 11:20
# @Description:
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """
    两次卷积
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # DropBlock(7, 0.9),
            nn.Dropout(0.8),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # DropBlock(7, 0.9),
            nn.Dropout(0.8),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU()
        )


# class Conv(nn.Sequential):
#     """
#     一次卷积
#     """
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         if mid_channels is None:
#             mid_channels = out_channels
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             # DropBlock(7, 0.9),
#             nn.Dropout(0.8),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        # 原论文采用的是转置卷积，我们一般用双线性插值
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 采用转置卷积的通道数会减少一半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1:卷积后的
        x2:上层跳跃连接过来的
        """
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 为了防止maxpooling后得到的图片尺寸向下取整，不是整数倍
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            #             nn.Sigmoid(),
        )


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, 1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        assert len(x.shape) == 4, f"好像乘不了"
        return x


class LGA_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 ):
        super(LGA_UNet, self).__init__()
        base_c = 16
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = DoubleConv(in_channels, out_channels=base_c, )  # (C, H, W)->(base_c, H, W)
        self.down1 = Down(base_c, base_c * 2)  # (base_c, H, W)->(base_c*2, H/2, W/2)
        self.down2 = Down(base_c * 2, base_c * 4)  # (base_c*2, H/2, W/2)->(base_c*4, H/4, W/4)
        self.down3 = Down(base_c * 4, base_c * 8)  # (base_c*4, H/4, W/4)->(base_c*8, H/8, W/8)

        self.conv2 = nn.Sequential(nn.Conv2d(base_c * 8, base_c * 8, kernel_size=3, padding=1, bias=False),
                                   # DropBlock(7, 0.9),
                                   nn.BatchNorm2d(base_c * 8),
                                   nn.ReLU(inplace=True))
        self.att1 = Attention()
        self.up1 = Up(base_c * 8, base_c * 4)
        self.up2 = Up(base_c * 4, base_c * 2)
        self.up3 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)  # down1
        x3 = self.down2(x2)  # down2
        x4 = self.down3(x3)  # down3
        x5 = self.att1(x4)     # up1
        x6 = self.conv2(x5)  # down4
        x = self.up1(x6, x3)  # up3
        x = self.up2(x, x2)  # up2
        x = self.up3(x, x1)  # up1
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    net = LGA_UNet(in_channels=2)
    summary(net, (2, 640, 640), device="cpu")
