from distutils.command.config import config
import torch
import yaml
from configuration.config import parse_arguments_bise
from data_io.ixi import IXI
from data_io.brats import BraTS2021
from torch.utils.data import DataLoader
from tools.utilize import *
from model.FT.fourier_transform import * 
from model.FT.power_spectrum import *
from metrics.kaid.stats import *
import numpy as np
from model.ae.kaid_ae import *
from loss_function.simclr_loss import *
from loss_function.common import cosine_similiarity
from loss_function.triplet_loss import triplet_loss
from model.cyclegan.cyclegan import CycleGen 
from model.munit.munit import Encoder as MUE
from model.munit.munit import Decoder as MUD
from model.unit.unit import Encoder as UE 


if __name__ == '__main__':
    args = parse_arguments_bise()
    with open('./configuration/kid/kid_{}.yaml'.format(args.dataset), 'r') as f:
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

    elif para_dict['noise_type'] == 'slight':
        noise_transform = [{'degrees': para_dict["a_rotation_degrees"],
                            'translate': [para_dict['a_trans_lower_limit'], para_dict['a_trans_upper_limit']],
                            'scale': [para_dict['a_scale_lower_limit'], para_dict['a_scale_upper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0},
                           {'degrees': para_dict['b_rotation_degrees'],
                            'translate': [para_dict['b_trans_lower_limit'], para_dict['b_trans_upper_limit']],
                            'scale': [para_dict['b_scale_lower_limit'], para_dict['b_scale_uppper_limit']],
                            'size': (para_dict['size'], para_dict['size']),'fillcolor': 0}]

    elif para_dict['noise_type'] == 'slight':
        noise_transform = [{'degrees':para_dict['severe_rotation'], 
                            'translate':[para_dict['severe_translation'], para_dict['severe_translation']],
                            'scale':[1-para_dict['severe_scaling'], 1+para_dict['severe_scaling']], 
                            'size':(para_dict['size'], para_dict['size'])},
                            {'degrees':para_dict['severe_rotation'], 
                             'translate':[para_dict['severe_translation'], para_dict['severe_translation']],
                             'scale':[1-para_dict['severe_scaling'], 1+para_dict['severe_scaling']], 
                             'size':(para_dict['size'], para_dict['size'])}]
    else:
        raise NotImplementedError('New Noise Has Not Been Implemented')
    
    #Dataset IO
    if para_dict['dataset'] == 'ixi':
        assert para_dict['source_domain'] in ['t2', 'pd']
        assert para_dict['target_domain'] in ['t2', 'pd']
    
        ixi_normal_dataset = IXI(root=para_dict['data_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type='normal',
                                 learn_mode='test', #train or test is meaningless
                                 transform_data=normal_transform,
                                 data_mode='paired',
                                 data_paired_weight=1.0,
                                 data_splited=False)
        
        ixi_noise_dataset = IXI(root=para_dict['data_path'],
                                modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                noise_type=para_dict['noise_type'],
                                learn_mode='test', #train or test is meaningless
                                transform_data=noise_transform,
                                data_paired=True,
                                data_paired_weight=1.0,
                                data_splited=False)
    
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
                                         learn_mode='train', # train or test is meaningless
                                         transform_data=normal_transform,
                                         data_paired=True,
                                         data_paired_weight=1.0,
                                         data_splited=False)

        brats_noise_dataset = BraTS2021(root=para_dict['data_path'],
                                        modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                        noise_type=para_dict['noise_type'],
                                        learn_mode='train',
                                        extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                        transform_data=noise_transform,
                                        data_paired=True,
                                        data_paired_weight=1.0,
                                        data_spilited=False)

    else:
        raise NotImplementedError("New Data Has Not Been Implemented")

    #TODO: make sure normal and nosiy loader release the same order of dataset
    if para_dict['dataset'] == 'ixi':
        ixi_normal_loader = DataLoader(ixi_normal_dataset, num_workers=para_dict['num_workers'],
                                       batch_size=para_dict['batch_size'], shuffle=True)
        ixi_noisy_loader = DataLoader(ixi_noise_dataset, num_workers=para_dict['num_workers'],
                                      batch_size=para_dict['batch_size'], shuffle=True)
    if para_dict['dataset'] == 'brats2021':
        brats_normal_loader = DataLoader(brats_normal_dataset, num_workers=para_dict['num_workers'],
                                         batch_size=para_dict['batch_size'], shuffle=True)
        brats_noisy_loader = DataLoader(brats_noise_dataset, num_workers=para_dict['num_workers'],
                                        batch_size=para_dict['batch_size'], shuffle=True)
    
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
    
    # Model
    kaid_ae = KAIDAE().to(device)
    # Loss
    criterion_recon = torch.nn.L1Loss().to(device)
    #TODO: Triplet Loss Function added

    # Optimizer
    optimizer = torch.optim.Adam(kaid_ae.parameters(), lr=para_dict['lr'],
                                 betas=[para_dict['beta1'], para_dict['beta2']])

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=para_dict['step_size'],
                                                   gamma=para_dict['gamma']) 
    
    if para_dict['dataset'] == 'ixi':
        """
        IXI: PD and T2
        BraTS: T1, T2 and FLAIR
        """
        normal_loader = ixi_normal_loader 
        noisy_loader = ixi_noisy_loader
        assert para_dict['source_domain'] in ['pd', 't2']
        assert para_dict['target_domain'] in ['pd', 't2']
    elif para_dict['dataset'] == 'brats2021':
        normal_loader = brats_normal_loader 
        noisy_loader = brats_noisy_loader
        assert para_dict['source_domain'] in ['t1', 't2', 'flair']
        assert para_dict['target_domain'] in ['t1', 't2', 'flair']
    else:
        raise NotImplementedError('New Dataset Has Not Been Implemented Yet')
        
    # Debug Mode
    if para_dict['debug']:
        batch_limit = 2
    else:
        batch_limit = int(para_dict['pair_num'] / para_dict['batch_size'])

    # Training 
    for epoch in range(para_dict['num_epochs']):
        for i, batch in enumerate(normal_loader): 
        #TODO: noisy loader

            if i > batch_limit:
                break

            real_a = batch[para_dict['source_domain']]
            real_b = batch[para_dict['target_domain']]

            # Fourier Transform 
            real_a_kspace = torch_fft(real_a)
            real_b_kspace = torch_fft(real_b)

            real_a_hf = torch_high_pass_filter(real_a_kspace, beta_a)
            real_b_hf = torch_high_pass_filter(real_b_kspace, beta_b)

            real_a_lf = torch_low_pass_filter(real_a_kspace, beta_a)
            real_b_lf = torch_low_pass_filter(real_b_kspace, beta_b)

            """
            Magnitude: sqrt(re^2 + im^2) tells you the amplitude of the component at the corresponding frequency
            Phase: atan2(im, re) tells you the relative phase of that component
            """

            real_a_hf_mag = torch.abs(real_a_hf).to(device)
            real_a_lf_mag = torch.abs(real_a_lf).to(device)

            real_b_hf_mag = torch.abs(real_b_hf).to(device)
            real_b_lf_mag = torch.abs(real_b_lf).to(device)

            optimizer.zero_grad()

            real_a_hf_z, real_a_hf_hat = kaid_ae(real_a_hf_mag)
            real_a_lf_z, real_a_lf_hat = kaid_ae(real_a_lf_mag)

            real_b_hf_z, real_b_hf_hat = kaid_ae(real_b_hf_mag)
            real_b_lf_z, real_b_lf_hat = kaid_ae(real_b_lf_mag)

            """
            Reconstruction
            """
            loss_recon_real_a_hf = criterion_recon(real_a_hf_mag, real_a_hf_hat) 
            loss_recon_real_b_hf = criterion_recon(real_b_hf_mag, real_b_hf_hat)

            loss_recon_real_a_lf = criterion_recon(real_a_lf_mag, real_a_lf_hat) 
            loss_recon_real_b_lf = criterion_recon(real_b_lf_mag, real_b_lf_hat)

            loss_recon = (loss_recon_real_a_hf + loss_recon_real_b_hf 
                                + loss_recon_real_a_lf + loss_recon_real_b_lf)

            """
            Triplet Loss
            """
            kid_triplet_loss = triplet_loss()
            loss_total = kid_triplet_loss + loss_recon

            loss_total.backward()
            optimizer.step()
            lr_scheduler.step()

            # Print Log
            infor = '\r{}[Batch {}/{}] [Recons loss: {:.4f}] [Triplet loss: {:.4f}]'.format(
                        '', i, batch_limit, loss_recon.item(), kid_triplet_loss.item())
    
    #Prediction