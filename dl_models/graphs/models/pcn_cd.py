import torch
import torch.nn as nn

class PCN(nn.Module):
    def __init__(self, config):
        super(PCN, self).__init__()
        self.config = config
        

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def forward(self, x):
        pass




if __name__ == '__main__':
    """Network Test"""
    config = json.load(open('../../configs/pcn.json'))
    config = edict(config)
    
    #inp = torch.autograd.Variable(torch.randn(1, 200, 1, 1, 1))
    #print("input shape:", inp.shape)
    #g_net = Generator(config)
    #out = g_net(inp)
    #print("output shape:", out.shape)
