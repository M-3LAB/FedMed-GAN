import torch
import torch.nn as nn

__all__ = ['minibatch_stddev_layer', 'l1_diff', 'l2_diff']

def minibatch_stddev_layer(input, stddev_group=4, stddev_feat=1):
    batch, channel, height, width = input.shape
    group = min(batch, stddev_group)
    stddev = input.view(
        group, -1, stddev_feat, channel // stddev_feat, height, width
    )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True)
    stddev = stddev.mean(2)
    stddev = stddev.repeat(group, 1, height, width)

    return torch.cat([input, stddev], 1)

def l1_diff(z_1, z_2):
    """
    L1 Difference

    Args:
        z_1 (_type_): _description_
        z_2 (_type_): _description_
    """
    pass 

def l2_diff(z_1, z_2):
    """
    L2 Difference

    Args:
        z_1 (_type_): _description_
        z_2 (_type_): _description_
    """
    pass 
    