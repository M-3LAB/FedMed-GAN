import torch
import torch.nn as nn

__all__ = ['cosine_similarity', 'l1_diff', 'l2_diff', 'euclidean_distance']

def cosine_similiarity(real_z, fake_z):
    """
    Cosine Similiarity 

    Args:
        real_z (vector): the hidden space of real image
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l1 difference between two hidden space 
    
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = cos(real_z, fake_z)
    return distance

def l1_diff(real_z, fake_z):
    """
    L1 Difference

    Args:
        real_z (vector): the hidden space of real image
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l1 difference between two hidden space 
    
    """
    distance = torch.abs(real_z, fake_z)
    return distance

def l2_diff(real_z, fake_z):
    """
    L2 Difference

    Args:
        real_z (vector): the hidden space of real image 
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l2 difference between two hidden space
    """
    diff_tensor = real_z - fake_z
    distance = torch.norm(diff_tensor, p=2, dim=0, keepdim=True)
    return distance

def euclidean_distance(real_z, fake_z):
    """
    Euclidean Distance

    Args:
        real_z (vector): the hidden space of real image 
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the euclidean difference between two hidden space
    """
    pass
