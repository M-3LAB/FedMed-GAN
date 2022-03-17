import torch
import torch.nn as nn

__all__ = ['ExNNUnet']

class Ex2DUnet(nn.Module):
    def __init__(self):
        super(Ex2DUnet).__init__()
    
    def forward(self, x):
        pass