#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Project    : neural-holography
# @FileName   : ONet.py
# @Author     : Spring
# @Time       : 2023/6/13 20:33
# @Description:


# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF


# Summary
# from torchsummary import summary


# Onet model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)))


class ONET(nn.Module):
    def mp(self, x, kernel=2, stride=2):
        return F.max_pool2d(x, kernel, stride)

    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            features=[64, 32]
    ):
        super(ONET, self).__init__()
        self.top_en = nn.ModuleList()
        self.top_de = nn.ModuleList()
        self.btm_en = nn.ModuleList()
        self.btm_de = nn.ModuleList()

        # self.ups = nn.ModuleList()
        # self.downs = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Right part of ONET
        for feature in features:
            self.top_en.append(DoubleConv(in_channels, feature))
            self.top_en.append(nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))

            self.btm_en.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Left part of ONET
        for feature in reversed(features):
            self.top_de.append(nn.Conv2d(feature // 2, feature, 3, 1, 1))
            self.top_de.append(DoubleConv(feature * 2, feature))

            self.btm_de.append(nn.ConvTranspose2d(feature // 2, feature, kernel_size=2, stride=2))
            self.btm_de.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.top_bottleneck = DoubleConv(features[-1], features[-1] // 2)
        self.btm_bottleneck = DoubleConv(features[-1], features[-1] // 2)

        # Output
        self.final_conv = nn.Conv2d(features[0] * 2, out_channels, kernel_size=1)

    def forward(self, x):
        b, t = x, x
        top_skip_connections = []
        btm_skip_connections = []

        # Bottom part of the ONET

        for down in self.btm_en:
            b = down(b)
            btm_skip_connections.append(b)
            b = self.mp(b)

        b = self.btm_bottleneck(b)
        btm_skip_connections = btm_skip_connections[::-1]

        for idx in range(0, len(self.btm_de), 2):
            b = self.btm_de[idx](b)
            skip_connection = btm_skip_connections[idx // 2]
            if b.shape != skip_connection.shape:
                b = TF.resize(b, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, b), dim=1)
            b = self.btm_de[idx + 1](concat_skip)
        # Top part of the ONET
        save = False
        for idx, up in enumerate(self.top_en):
            t = up(t)
            # print(idx, t.shape)
            if idx % 2 != 0:
                # print('saved:', idx)
                top_skip_connections.append(t)
            save = not save
        t = self.top_bottleneck(t)
        # decoder

        top_skip_connections = top_skip_connections[::-1]

        for idx in range(0, len(self.top_de), 2):
            t = self.top_de[idx](t)
            skip_connection = top_skip_connections[idx // 2]
            # print('t', idx, t.shape)
            # print('sc:', idx//2, skip_connection.shape)

            if t.shape != skip_connection.shape:
                # print('resizing')
                t = TF.resize(t, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, t), dim=1)
            # print('concat: ', concat_skip.shape)
            t = self.mp(self.top_de[idx + 1](concat_skip))

            # print('ft:', idx, t.shape)
        # print(t.shape, b.shape, 'suuup')
        x = torch.cat((t, b), dim=1)
        return self.final_conv(x)


def test_model_o():
    x = torch.randn((1, 2, 512, 512)).to('cpu')
    model = ONET(in_channels=2, features=[32, 16]).to('cpu')
    preds = model(x)
    print('input ', preds.shape)
    print('output: ', x.shape)
    # summary(model, input_size=(1, 512, 512), device='cpu')
    from torchinfo import summary
    summary(model, input_size=(1, 2, 1080, 1920))
    # assert preds.shape == x.shape


if __name__ == '__main__':
    test_model_o()


# ==========================================================================================
# Total params: 120,465
# Trainable params: 120,465
# Non-trainable params: 0
# Total mult-adds (G): 829.34
# ==========================================================================================
# Input size (MB): 16.59
# Forward/backward pass size (MB): 58508.70
# Params size (MB): 0.48
# Estimated Total Size (MB): 58525.77
# ==========================================================================================