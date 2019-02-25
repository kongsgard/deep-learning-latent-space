import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict

class PCNDecoder(nn.Module):
    def __init__(self, config):
        super(PCNDecoder, self).__init__()
        self.config = config
        self.num_fine = self.config.grid_size ** 2 * self.config.num_coarse

        self.coarse_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.config.num_coarse * 3),
            nn.LeakyReLU(),
        )

        self.fine_layer = nn.Sequential(
            nn.Conv1d(1029, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 3, kernel_size=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        coarse = self.coarse_layer(x)
        coarse = coarse.view(-1, self.config.num_coarse, 3) # [32, 1024, 3]

        grid_features = torch.meshgrid([
            torch.linspace(-self.config.grid_scale, self.config.grid_scale, self.config.grid_size),
            torch.linspace(-self.config.grid_scale, self.config.grid_scale, self.config.grid_size)
        ]) # TODO: Might flip axes according to reference
        grid_features = torch.reshape(torch.stack(grid_features, dim=2), (-1, 2)).expand(1, -1, -1) # TODO: Change .expand to .view?
        grid_features = grid_features.repeat(x.shape[0], self.config.num_coarse, 1) # [32, 16384, 2]

        point_features = coarse.view(-1, coarse.shape[1], 1, 3)
        point_features = point_features.repeat(1, 1, self.config.grid_size ** 2, 1)
        point_features = point_features.view(-1, self.num_fine, 3) # [32, 16384, 3]

        global_features = x.view(-1, 1, x.shape[1])
        global_features = global_features.repeat(1, self.num_fine, 1) # [32, 16384, 1024]

        features = torch.cat((grid_features, point_features, global_features), 2) # [32, 16384, 1029]
        features = features.transpose(2, 1) # [32, 1029, 16384]

        center = coarse.view(-1, coarse.shape[1], 1, 3)
        center = center.repeat(1, 1, self.config.grid_size ** 2, 1)
        center = center.view(-1, self.num_fine, 3) # [32, 16384, 3]
        center = center.transpose(2, 1) # [32, 3, 16384]

        fine = self.fine_layer(features) + center # [32, 3, 16384]
        return coarse, fine.transpose(2, 1)


if __name__ == '__main__':
    """Network Test"""
    config = json.load(open('../../configs/pcn.json'))
    config = edict(config)
    
    x_global = torch.autograd.Variable(torch.randn(32, 1024))
    network = PCNDecoder(config)
    coarse, fine = network(x_global)
    print(coarse.shape)
    print(fine.shape)
