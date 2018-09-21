import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.config = config
        self.cube_len = self.config.cube_len

        padding = (0, 0, 0)
        if self.cube_len == 32:
            padding = (1, 1, 1)

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=self.config.bias, padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len),
            nn.LeakyReLU(self.config.leak_value)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(self.cube_len, self.cube_len * 2, kernel_size=4, stride=2, bias=self.config.bias,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len * 2),
            nn.LeakyReLU(self.config.leak_value)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(self.cube_len * 2, self.cube_len * 4, kernel_size=4, stride=2, bias=self.config.bias,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len * 4),
            nn.LeakyReLU(self.config.leak_value)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(self.cube_len * 4, self.cube_len * 8, kernel_size=4, stride=2, bias=self.config.bias,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len * 8),
            nn.LeakyReLU(self.config.leak_value)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(self.cube_len * 8, 1, kernel_size=4, stride=2, bias=self.config.bias, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, self.config.cube_len, self.config.cube_len, self.config.cube_len)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
        return out


def main():
    """Discriminator Test"""
    config = json.load(open('../../configs/3dgan_exp_0.json'))
    config = edict(config)
    inp = torch.autograd.Variable(torch.randn(1, 20, 64, 64, 64))
    print("input shape:", inp.shape)
    d_net = Discriminator(config)
    out = d_net(inp)
    print("output shape:", out.shape)


if __name__ == '__main__':
    main()
