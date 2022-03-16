import numpy as np
import torch
from model.FT.power_spectrum import *
from model.FT.fourier_transform import *
from tools.utilize import *

__all__ = ['mask_stats', 'frequency_diff', 'mask_frequency_diff', 'delta_diff',
           'best_msl_list']

def frequency_diff(hf_kspace, lf_kspace):
    """
    hf is the shortname of High Frequency 
    lf is the shortname of Low Frequency
    """
    hf_kspace_abs = torch.abs(hf_kspace) 
    lf_kspace_abs = torch.abs(lf_kspace) 
    hf_total = torch.sum(hf_kspace_abs)
    lf_total = torch.sum(lf_kspace_abs)
    diff = torch.abs(hf_total - lf_total).numpy() 
    return diff

def mask_frequency_diff(kspace, msl):

    diff_list = []

    hf = torch_high_pass_filter(kspace, msl)
    lf = torch_low_pass_filter(kspace, msl)

    batchsize = kspace.size()[0]
    for idx in range(batchsize):
        diff = frequency_diff(hf[idx,:,:,:], lf[idx,:,:,:])
        diff_list.append(diff)
    
    return diff_list

def delta_diff(kspace, msl_init, img_size):
    msl_max_limit = img_size / 4

    diff_list = mask_frequency_diff(kspace, msl_init)
    #print(f'msl_init: {msl_init}')
    #print(f'diff_list init: {diff_list}')

    avg_diff = average(diff_list)
    print(f'average_diff init: {avg_diff}')

    msl = msl_init 

    delta_dic = {} 

    for _ in range(100):

        msl += 1
        if msl > msl_max_limit:
            break

        new_diff_list = mask_frequency_diff(kspace, msl)
        #print(f'msl: {msl}')
        #print(f'new diff_list: {new_diff_list}')
        new_avg_diff = average(new_diff_list)
        #print(f'new average_diff : {new_avg_diff}')

        delta = np.abs(new_avg_diff - avg_diff) 
        delta_dic[msl] = delta

        if new_avg_diff < avg_diff:
            avg_diff = new_avg_diff

    return delta_dic
        
def mask_stats(data_loader, source_domain, target_domain, src_msl=None, tag_msl=None, img_size=256):

    """
    Args:
        msl: the half of mask side length  
        src: source domain 
        tag: target domain
    """
    
    if src_msl is None:
        src_msl = 2

    if tag_msl is None:
        tag_msl = 2
    
    real_a_list = torch.randn(data_loader.batch_size, 1, img_size, img_size)
    real_b_list = torch.randn(data_loader.batch_size, 1, img_size, img_size)

    for i, batch in enumerate(data_loader):
        # two modality: source domain(A) and target domain(B). The aim is to generate B image 
        real_a = batch[source_domain]
        real_b = batch[target_domain]
        real_a_list = concate_tensor_lists(real_a_list, real_a, i) 
        real_b_list = concate_tensor_lists(real_b_list, real_b, i) 

    print(f'loop concate finished')
    real_a_kspace = torch_fft(real_a_list) 
    real_b_kspace = torch_fft(real_b_list) 

    #print(f'src_msl: {src_msl}')
    #print(f'tag_msl: {tag_msl}')
    a_delta_dic = delta_diff(real_a_kspace, src_msl, img_size)
    b_delta_dic = delta_diff(real_b_kspace, tag_msl, img_size)

    return a_delta_dic, b_delta_dic

def best_msl_list(delta_dic, delta_diff=None):

    #print(f'delta_diff: {delta_diff}')

    msl_list = []
    for key in delta_dic:
        if delta_dic[key] < delta_diff: 
            msl_list.append(key)

    return msl_list




        
    
    

