import torch
import torch.nn as nn

__all__ = ['cosine_similarity', 'l1_diff', 'l2_diff', 'euclidean_distance']

def cosine_similiarity(t1, t2):
    """
    Args:
        t1 (torch.tensor): first feacture  
        t2 (torch.tensor): second feacture 
    """
    pass

def l1_diff(real_z, fake_z):
    """
    L1 Difference

    Args:
        real_z (vector): the hidden space of real image
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l1 difference between two hidden space 
    
    """
    pass 

def l2_diff(real_z, fake_z):
    """
    L2 Difference

    Args:
        real_z (vector): the hidden space of real image 
        fake_z (vector): the hidden space of fake image 
    
    Output:
        the l2 difference between two hidden space
    """
    pass 

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
