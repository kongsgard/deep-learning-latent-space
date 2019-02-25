import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict

class PCNEncoder(nn.Module):
    def __init__(self, config):
        super(PCNEncoder, self).__init__()
        self.config = config
        
        self.layer1 = nn.Sequential(
                nn.Conv1d(3, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=1),
                nn.ReLU(),
            )

        self.layer2 = nn.Sequential(
                nn.Conv1d(512, 1024, kernel_size=1),
                nn.ReLU(),
            ) # TODO: Add second layer

    def forward(self, x):
        x = self.layer1(x) # [32, 256, 2048]

        x_global = torch.max(x, 2, keepdim=True)[0] # [32, 256, 1]
        x_global_repeated = x_global.repeat(1, 1, x.shape[2]) # [32, 256, 2048]

        x = torch.cat((x, x_global_repeated), 1) # [32, 512, 2048] 
        x = self.layer2(x) # [32, 1024, 2048]
        x = torch.max(x, 2)[0] # [32, 1024]
        return x


if __name__ == '__main__':
    """Network Test"""
    config = json.load(open('../../configs/pcn.json'))
    config = edict(config)
    
    input_points = torch.autograd.Variable(torch.randn(32, 2048, 3))
    input_points = input_points.transpose(2, 1)
    network = PCNEncoder(config) 
    out = network(input_points)
    print(out)
