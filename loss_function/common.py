import torch
import torch.nn as nn

__all__ = ['cosine_similarity', 'l1_diff', 'l2_diff']

def cosine_similiarity(t1, t2):
    """
    Args:
        t1 (torch.tensor): first feacture  
        t2 (torch.tensor): second feacture 
    """
    pass

def l1_diff(z_1, z_2):
    """
    L1 Difference

    Args:
        z_1 (vector): the hidden space of real image
        z_2 (vector): the hidden space of fake image 
    
    Output:
        the l1 difference between two hidden space 
    
    """
    pass 

def l2_diff(z_1, z_2):
    """
    L2 Difference

    Args:
        z_1 (vector): the hidden space of real image 
        z_2 (vector): the hidden space of fake image 
    
    Output:
        the l2 difference between two hidden space
    """
    pass 
