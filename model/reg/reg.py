import torch
import torch.nn as nn
from model.reg.modules import ResUnet

__all__ = ['Reg']

class Reg(nn.Module):
    def __init__(self, img_size, device):
        super(Reg, self).__init__()

        init_func = 'kaiming'
        init_to_identity = True

        self.oh = img_size 
        self.ow = img_size 

        self.in_channels_a = 1
        self.in_channels_b = 1

        self.device = device
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg='A',
                                  init_func=init_func, init_to_identity=init_to_identity).to(self.device)
        
        self.identity_grid = self.get_identity_grid()
    
    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity
    
    def forward(self, img_a, img_b):
        deformations = self.offset_map(img_a, img_b)
        return deformations 


