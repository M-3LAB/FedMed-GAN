import torch

__all__ = ['smoothing_loss']

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    #grad = d 
    return d