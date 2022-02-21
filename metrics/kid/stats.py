import numpy as np
import torch
from model.FT.power_spectrum import *
from model.FT.fourier_transform import *
from evaluation.common import *

__all__ = ['beta_stats', 'frequency_diff', 'beta_frequency_diff', 'delta_diff',
            'best_beta_list']

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

def beta_frequency_diff(kspace, beta):

    diff_list = []

    kspace_hf = torch_high_pass_filter(kspace, beta)
    kspace_lf = torch_low_pass_filter(kspace, beta)

    # batchsize = kspace.size()[0]
    for idx in range(kspace.size()[0]):
        diff = frequency_diff(kspace_hf[idx,:,:,:] - kspace_lf[idx,:,:,:])
        diff_list.append(diff)
    
    return diff_list

def delta_diff(kspace, beta_init):
    img_size = kspace.size()[2]
    beta_max_limit = img_size / 4

    diff_list = beta_frequency_diff(kspace, beta_init)
    avg_diff = average(diff_list)
    beta = beta_init 

    delta_dic = {} 
    #diff_dic = {}

    for _ in range(100):

        beta += 1
        if beta > beta_max_limit:
            break

        new_diff_list = beta_frequency_diff(kspace, beta)
        new_avg_diff = average(new_diff_list)

        if new_avg_diff < avg_diff:
            avg_diff = new_avg_diff

        delta = np.abs(new_avg_diff - avg_diff) 
        delta_dic[beta] = delta

    return delta_dic
        
def beta_stats(data_loader, source_domain, target_domain, beta_a=None, 
               beta_b=None, img_size=256):
    """
    Args:
        beta:  (2 * beta) **2 is the size of the mask
    """
    
    if beta_a is None:
        beta_a = 1

    if beta_b is None:
        beta_b = 1
    
    real_a_list = torch.randn(data_loader.batch_size, 1, img_size, img_size)
    real_b_list = torch.randn(data_loader.batch_size, 1, img_size, img_size)

    #result_dic = {} # The first element is domain a and the second element is domain b

    for i, batch in enumerate(data_loader):
        # two modality: source domain(A) and target domain(B). The aim is to generate B image 
        real_a = batch[source_domain]
        real_b = batch[target_domain]
        real_a_list = concate_tensor_lists(real_a_list, real_a, i) 
        real_b_list = concate_tensor_lists(real_b_list, real_b, i) 

    real_a_kspace = torch_fft(real_a_list) 
    real_b_kspace = torch_fft(real_b_list) 

    a_delta_dic = delta_diff(real_a_kspace, beta_a)
    b_delta_dic = delta_diff(real_b_kspace, beta_b)

    return a_delta_dic, b_delta_dic

def best_beta_list(delta_dic, delta_diff):

    beta_list = []
    for key in delta_dic:
        if delta_dic[key] < delta_diff: 
            beta_list.append(key)

    return beta_list




        
    
    

