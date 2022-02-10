import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Reg_Transformer']

class Reg_Transformer(nn.Module):
    def __init__(self):
        super(Reg_Transformer, self).__init__()

    def forward(self,src, flow, device):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).to(device)
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        return warped