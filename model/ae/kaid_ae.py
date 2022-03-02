import torch
import torch.nn as nn
from model.cyclegan.cyclegan import UNetDown, UNetUp

__all__ = ['KAIDAE']

class KAIDAE(nn.Module):

    def __init__(self):
        super(KAIDAE, self).__init__()
        self.down1 = UNetDown(1, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64) 
        self.up5 = UNetUp(64, 1)
    
    def encode(self, x):
        self.d1 = self.down1(x)
        self.d2 = self.down2(self.d1)
        self.d3 = self.down3(self.d2)
        self.d4 = self.down4(self.d3)
        self.z = self.down5(self.d4)
        return self.z 

    def decode(self, z=None):
        z_in = self.z if z is None else z 
        self.u1 = self.up1(z_in)
        self.u2 = self.up2(self.u1, self.d4)
        self.u3 = self.up3(self.u2, self.d3)
        self.u4 = self.up4(self.u3, self.d2)
        self.x_hat = self.up5(self.u4, self.d1)
        return self.x_hat 

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

