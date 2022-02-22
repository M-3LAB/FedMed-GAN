import torch
import torch.nn as nn
from model.cyclegan.cyclegan import UNetDown, UNetUp

__all__ = ['KIDAE']

class KIDAE(nn.Module):

    def __init__(self):
        super(KIDAE, self).__init__()
        self.down1 = UNetDown(1, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64) 
    
    def encode(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        self.z = self.down5(d4)
        return self.z 

    def decode(self, z=None):
        z_in = self.z if z is None else z 
        pass

    def forward(self, x, phase='encoding'):
        assert phase in ['encoding', 'decoding'], 'phase should be in decoding and encoding'
        pass

