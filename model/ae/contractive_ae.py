import torch
import toch.nn as nn
from model.cyclegan import *

__all__ = ['CAE']

class CAE(nn.Module):

    def __init__(self, inc, z_dim):
        super(CAE, self).__init__()
        self.sigmoid = nn.Sigmoid()
        pass
    
    def encoder(self, x):
        pass

    def decoder(self, z):
        pass

    def forward(self, x):
        pass
