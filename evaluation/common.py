import torch

__all__ = ['average', 'concate_tensor_lists']

def average(l):
   return sum(l) / len(l)  

def concate_tensor_lists(imgs_list, img, i):
    if i == 0:
        imgs_list = img
    else: 
        imgs_list = torch.cat((imgs_list, img), 0)
    return imgs_list