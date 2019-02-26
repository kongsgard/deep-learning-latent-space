import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict

class PCN(nn.Module):
    def __init__(self, config):
        super(PCN, self).__init__()
        self.config = config
        self.num_fine = self.config.grid_size ** 2 * self.config.num_coarse

        self.encoder_layer1 = nn.Sequential(
                nn.Conv1d(3, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=1),
                nn.ReLU(),
            )

        self.encoder_layer2 = nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(512, 1024, kernel_size=1),
                nn.ReLU(),
            )

        self.decoder_coarse_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.config.num_coarse * 3),
            nn.LeakyReLU(),
        )

        self.decoder_fine_layer = nn.Sequential(
            nn.Conv1d(1029, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 3, kernel_size=1),
            nn.LeakyReLU(),
        )

    def encode(self, x):
        # The Conv1d operator requires an input with shape (N, C_in, L_in) which necessitates a transpose of the input_points
        # Perform the inverse transpose of the network output
        x = x.transpose(2, 1)
        x = self.encoder_layer1(x) # [32, 256, 2048]

        x_global = torch.max(x, 2, keepdim=True)[0] # [32, 256, 1]
        x_global_repeated = x_global.repeat(1, 1, x.shape[2]) # [32, 256, 2048]

        x = torch.cat((x, x_global_repeated), 1) # [32, 512, 2048] 
        x = self.encoder_layer2(x) # [32, 1024, 2048]
        x = torch.max(x, 2)[0] # [32, 1024]
        return x

    def decode(self, x):
        coarse = self.decoder_coarse_layer(x)
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
        
        center = coarse.view(-1, coarse.shape[1], 1, 3)
        center = center.repeat(1, 1, self.config.grid_size ** 2, 1)
        center = center.view(-1, self.num_fine, 3) # [32, 16384, 3]
        
        # The Conv1d operator requires an input with shape (N, C_in, L_in) which necessitates a transpose of the input_points
        # Perform the inverse transpose of the network output
        features = features.transpose(2, 1) # [32, 1029, 16384]
        center = center.transpose(2, 1) # [32, 3, 16384]
        fine = self.decoder_fine_layer(features) + center # [32, 3, 16384]
        return coarse, fine.transpose(2, 1)

    def forward(self, x):
        feature_global = self.encode(x)
        coarse, fine = self.decode(feature_global)
        return coarse, fine


if __name__ == '__main__':
    """Network Test"""
    config = json.load(open('../../configs/pcn.json'))
    config = edict(config)
    
    input_points = torch.autograd.Variable(torch.randn(32, 2048, 3))
    network = PCN(config)
    coarse, fine = network(input_points)
    print(coarse.shape)
    print(fine.shape)
