import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cube_len = self.config.cube_len

        padding = (0, 0, 0)
        if self.cube_len == 32:
            padding = (1, 1, 1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.config.z_size, self.cube_len * 8, kernel_size=4, stride=2, bias=self.config.bias,
                                     padding=padding),
            torch.nn.BatchNorm3d(self.cube_len * 8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 8, self.cube_len * 4, kernel_size=4, stride=2, bias=self.config.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len * 4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 4, self.cube_len * 2, kernel_size=4, stride=2, bias=self.config.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len * 2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 2, self.cube_len, kernel_size=4, stride=2, bias=self.config.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=self.config.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.config.z_size, 1, 1, 1)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([100, 1, 64, 64, 64])

        return out


def main():
    """Generator Test"""
    config = json.load(open('../../configs/3dgan_exp_0.json'))
    config = edict(config)
    inp = torch.autograd.Variable(torch.randn(1, 200, 1, 1, 1))
    print("input shape:", inp.shape)
    #print(inp.shape)
    g_net = Generator(config)
    out = g_net(inp)
    #print(out.shape)


if __name__ == '__main__':
    main()