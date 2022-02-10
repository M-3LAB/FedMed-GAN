import torch
import torch.nn as nn

__all__ = ['SSAE']

class SSAE(nn.Module):

    def __init__(self):
        super(SSAE, self).__init__()
        self.sigmoid = nn.Sigmoid()
        pass
    
    def encoder(self, x):
        pass

    def decoder(self, z):
        pass

    def forward(self, x):
        pass

