import torch
import yaml
from configuration.config import parse_arguments_bise
from data_io.ixi import IXI
from data_io.brats import BraTS2021
from torch.utils.data import DataLoader
from tools.utilize import *
from model.FT.fourier_transform import * 
from model.FT.power_spectrum import *
from metrics.kid.stats import *
import numpy as np
from model.ae.kid_ae import *

if __name__ == '__main__':
    args = parse_arguments_bise()
    with open('./configuration/bise/bise_{}.yaml'.format(args.dataset), 'r') as f:
        para_dict = yaml.load(f, Loader=yaml.SafeLoader)
    para_dict = merge_config(para_dict, args)
    print(para_dict)

    file_path = record_path(para_dict)
    if para_dict['save_log']:
        save_arg(para_dict, file_path)
        save_script(__file__, file_path)

    with open('./work_dir/log_running.txt'.format(file_path), 'a') as f:
        print('---> {}'.format(file_path), file=f)
        print(para_dict, file=f)

    device, device_ids = parse_device_list(para_dict['gpu_ids'], 
                                           int(para_dict['gpu_id'])) 

    seed_everything(para_dict['seed'])

    normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                         'scale':[1.00, 1.00], 
                         'size':(para_dict['size'], para_dict['size'])},
                        {'degrees':0, 'translate':[0.00, 0.00],
                         'scale':[1.00, 1.00], 
                         'size':(para_dict['size'], para_dict['size'])}]

    if para_dict['noise_type'] == 'gaussian':
        noise_transform = [{'mu':para_dict['a_mu'], 'sigma':para_dict['a_sigma'],
                            'size':(para_dict['size'], para_dict['size'])},
                           {'mu':para_dict['b_mu'], 'sigma':para_dict['b_sigma'],
                            'size':(para_dict['size'], para_dict['size'])}]

    elif para_dict['noise_type'] == 'reg':
        noise_transform = [{'degrees': para_dict["a_rotation_degrees"],
                            'translate': [para_dict['a_trans_lower_limit'], para_dict['a_trans_upper_limit']],
                            'scale': [para_dict['a_scale_lower_limit'], para_dict['a_scale_upper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0},
                           {'degrees': para_dict['b_rotation_degrees'],
                            'translate': [para_dict['b_trans_lower_limit'], para_dict['b_trans_upper_limit']],
                            'scale': [para_dict['b_scale_lower_limit'], para_dict['b_scale_uppper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0}]
    else:
        raise NotImplementedError('New Noise Has not been Implemented')
    
    #Dataset IO
    if para_dict['dataset'] == 'ixi':
        assert para_dict['source_domain'] in ['t2', 'pd']
        assert para_dict['target_domain'] in ['t2', 'pd']
    
        ixi_normal_dataset = IXI(root=para_dict['data_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type='normal',
                                 mode='test', #train or test is meaningless
                                 transform_data=normal_transform,
                                 paired=True,
                                 clients=[1.0],
                                 seperated=False,
                                 splited=False,
                                 regenerate_data=True)
        
        ixi_noise_dataset = IXI(root=para_dict['data_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type=para_dict['noise_type'],
                                 mode='test', #train or test is meaningless
                                 transform_data=noise_transform,
                                 paired=True,
                                 clients=[1.0],
                                 seperated=False,
                                 splited=False,
                                 regenerate_data=True) 
    
    elif para_dict['dataset'] == 'brats2021':
        assert para_dict['source_domain'] in ['t1', 't2', 'flair']
        assert para_dict['target_domain'] in ['t1', 't2', 'flair']

        """
        #TODO: Create a dataset contained the whole part of BraTS 2021, included training and validation 
        """
        
        brats_normal_dataset = BraTS2021(root=para_dict['data_path'],
                                         modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                         extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                         noise_type='normal',
                                         mode='train', # train or test is meaningless
                                         transform_data=normal_transform,
                                         paired=True,
                                         clients=1.0,
                                         seperated=False,
                                         regenerate_data=True)

        brats_noise_dataset = BraTS2021(root=para_dict['valid_path'],
                                        modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                        noise_type=para_dict['noise_type'],
                                        mode='train',
                                        extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                        transform_data=noise_transform,
                                        paired=True,
                                        seperated=False,
                                        regenerate_data=True)

    else:
        raise NotImplementedError("New Data has not been Implemented")

    #TODO: make sure normal and nosie loader release the same order of dataset
    if para_dict['dataset'] == 'ixi':
        ixi_normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                       batch_size=para_dict['batch_size'], shuffle=False)
        ixi_noise_loader = DataLoader(ixi_noise_dataset, num_workers=para_dict['num_workers'],
                                      batch_size=para_dict['batch_size'], shuffle=False)
    if para_dict['dataset'] == 'brats':
        brats_normal_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                         batch_size=para_dict['batch_size'], shuffle=False)
        brats_noise_loader = DataLoader(brats_noise_dataset, num_workers=para_dict['num_workers'],
                                        batch_size=para_dict['batch_size'], shuffle=False)
    
    if para_dict['debug']:
        batch_limit = 2

    # Beta Spectral Statistics  
    if para_dict['bise_stats']:
        src_dict, tag_dict = beta_stats(ixi_normal_loader, para_dict['source_domain'], 
                                        para_dict['target_domain'])
        print(f"source_domain: {para_dict['source_domain']}, its_dict: {src_dict}")
        print(f"target_domain: {para_dict['source_domain']}, its_dict: {src_dict}")

        src_best_beta_list = best_beta_list(src_dict)
        tag_best_beta_list = best_beta_list(tag_dict)

        beta_a = src_best_beta_list[0]
        beta_b = tag_best_beta_list[0]
    else:
        beta_a = np.load(para_dict['src_beta_init_path'])
        beta_b = np.load(para_dict['tag_beta_init_path'])
        
    # Fourier Transform 
    for i, batch in enumerate(ixi_normal_loader): 
        if i > batch_limit:
            break
        """
        IXI: PD and T2
        BraTS: T1, T2 and FLAIR
        """
        real_a = batch[para_dict['source_domain']]
        real_b = batch[para_dict['target_domain']]

        real_a_kspace = torch_fft(real_a)
        real_b_kspace = torch_fft(real_b)
        
        real_a_kspace_hf = torch_high_pass_filter(real_a_kspace, beta_a)
        real_b_kspace_hf = torch_high_pass_filter(real_b_kspace, beta_b)

        real_a_kspace_lf = torch_low_pass_filter(real_a_kspace, beta_a)
        real_b_kspace_lf = torch_low_pass_filter(real_b_kspace, beta_b)

        real_a_kspace_hf_abs = torch.abs(real_a_kspace_hf)
        real_a_kspace_lf_abs = torch.abs(real_a_kspace_lf)

        real_b_kspace_hf_abs = torch.abs(real_b_kspace_hf)
        real_b_kspace_lf_abs = torch.abs(real_b_kspace_lf)