from pyclbr import Class
import torch
import yaml
import os
import numpy as np

from tools.utilize import * 
from data_io.brats import BraTS2021
from data_io.ixi import IXI
from torch.utils.data import DataLoader
from tools.utilize import *
from tools.visualize import plot_sample

import warnings
warnings.filterwarnings("ignore")

class FederatedTrain():
    def __init__(self, args):
        self.args = args
        self.para_dict = None
        # data
        self.client_data_list = []

        # server and client model
        self.server = None
        self.clients = []
        self.device = None

        # global model
        self.server_gener = {'from_a_to_b': None, 'from_b_to_a': None}
        self.server_discr = {'from_a_to_b': None, 'from_b_to_a': None}
        self.client_gener_list = {'from_a_to_b': [], 'from_b_to_a': []} 
        self.client_discr_list = {'from_a_to_b': [], 'from_b_to_a': []}  
        self.client_psnr_list = []

        # save and log
        self.file_path = None
        # save model
        self.best_psnr = 0

    def load_config(self):
        with open('./configuration/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/2_train_base/federated_training.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)

        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)

        if not self.para_dict['contraD']:
            self.para_dict['num_augmentation'] == 'all'

    def preliminary(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

        assert self.para_dict['source_domain'] != self.para_dict['target_domain']
        seed_everything(self.para_dict['seed'])

        device, device_ids = parse_device_list(self.para_dict['gpu_ids'], int(self.para_dict['gpu_id']))
        self.device = torch.device("cuda", device)

        self.file_path = record_path(self.para_dict)
        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.file_path)
            save_script(__file__, self.file_path)

        self.fid_stats_from_a_to_b = '{}/{}/{}_{}_fid_stats.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['source_domain'], self.para_dict['target_domain'])
        self.fid_stats_from_b_to_a = '{}/{}/{}_{}_fid_stats.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['target_domain'], self.para_dict['source_domain'])

        if self.para_dict['fid']:
            if not os.path.exists(self.fid_stats_from_a_to_b):
                os.system(r'python3 fid_stats.py --dataset {} --source-domain {} --target-domain {} --gpu-id {} --valid-path {}'.format(
                    self.para_dict['dataset'], self.para_dict['source_domain'], self.para_dict['target_domain'], self.para_dict['gpu_id'], self.para_dict['valid_path']))
            if not os.path.exists(self.fid_stats_from_a_to_b):
                raise NotImplementedError('FID Still Not be Implemented Yet')
            else:
                print('fid stats from a to b: {}'.format(self.fid_stats_from_a_to_b))

            if not os.path.exists(self.fid_stats_from_b_to_a):
                os.system(r'python3 fid_stats.py --dataset {} --source-domain {} --target-domain {} --gpu-id {} --valid-path {}'.format(
                    self.para_dict['dataset'], self.para_dict['target_domain'], self.para_dict['source_domain'], self.para_dict['gpu_id'], self.para_dict['valid_path']))
            if not os.path.exists(self.fid_stats_from_b_to_a):
                raise NotImplementedError('FID Still Not be Implemented Yet')
            else:
                print('fid stats from b to a: {}'.format(self.fid_stats_from_b_to_a))

        print('work dir: {}'.format(self.file_path))
        print('---------------------')


    def load_data(self):
        self.normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                                     'scale':[1.00, 1.00], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'degrees':0, 'translate':[0.00, 0.00],
                                     'scale':[1.00, 1.00], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]
        self.gaussian_transform = [{'mu':self.para_dict['a_mu'], 'sigma':self.para_dict['a_sigma'],
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'mu':self.para_dict['b_mu'], 'sigma':self.para_dict['b_sigma'],
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]
        self.slight_transform = [{'degrees': self.para_dict['noise_level'], 
                                     'translate': [0.02*self.para_dict['noise_level'], 0.02*self.para_dict['noise_level']],
                                     'scale':[1-0.02*self.para_dict['noise_level'], 1-0.02*self.para_dict['noise_level']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'degrees': self.para_dict['noise_level'], 
                                     'translate': [0.02*self.para_dict['noise_level'], 0.02*self.para_dict['noise_level']],
                                     'scale':[1-0.02*self.para_dict['noise_level'], 1-0.02*self.para_dict['noise_level']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]
        self.severe_transform = [{'degrees':self.para_dict['severe_rotation'], 
                                     'translate':[self.para_dict['severe_translation'], self.para_dict['severe_translation']],
                                     'scale':[1-self.para_dict['severe_scaling'], 1+self.para_dict['severe_scaling']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                    {'degrees':self.para_dict['severe_rotation'], 
                                     'translate':[self.para_dict['severe_translation'], self.para_dict['severe_translation']],
                                     'scale':[1-self.para_dict['severe_scaling'], 1+self.para_dict['severe_scaling']], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])}]

        if self.para_dict['noise_type'] == 'normal':
            self.noise_transform = self.normal_transform
        elif self.para_dict['noise_type'] == 'gaussian':
            self.noise_transform = self.gaussian_transform
        elif self.para_dict['noise_type'] == 'slight':
            self.noise_transform = self.slight_transform
        elif self.para_dict['noise_type'] == 'severe': 
            self.noise_transform = self.severe_transform
        else:
            raise NotImplementedError('New Noise Has not been Implemented')

        if self.para_dict['dataset'] == 'brats2021':
            assert self.para_dict['source_domain'] in ['t1', 't2', 'flair']
            assert self.para_dict['target_domain'] in ['t1', 't2', 'flair']
            self.train_dataset = BraTS2021(root=self.para_dict['data_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           noise_type=self.para_dict['noise_type'],
                                           learn_mode='train',
                                           transform_data=self.noise_transform,
                                           client_weights=self.para_dict['clients_data_weight'],
                                           data_mode=self.para_dict['data_mode'],
                                           data_num=self.para_dict['data_num'],
                                           data_paired_weight=self.para_dict['data_paired_weight'],
                                           data_moda_ratio=self.para_dict['data_moda_ratio'],
                                           data_moda_case=self.para_dict['data_moda_case'])
            self.valid_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='normal',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.normal_transform,
                                           data_mode='paired')
            self.assigned_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='severe',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.severe_transform,
                                           data_mode='paired',
                                           assigned_data=self.para_dict['single_img_infer'],
                                           assigned_images=self.para_dict['assigned_images']) 
        elif self.para_dict['dataset'] == 'ixi':
            assert self.para_dict['source_domain'] in ['t2', 'pd']
            assert self.para_dict['target_domain'] in ['t2', 'pd']
            self.train_dataset = IXI(root=self.para_dict['data_path'],
                                    modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                    extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                    noise_type=self.para_dict['noise_type'],
                                    learn_mode='train',
                                    transform_data=self.noise_transform,
                                    data_mode=self.para_dict['data_mode'],
                                    data_num=self.para_dict['data_num'],
                                    data_paired_weight=self.para_dict['data_paired_weight'],
                                    client_weights=self.para_dict['clients_data_weight'],
                                    dataset_splited=True,
                                    data_moda_ratio=self.para_dict['data_moda_ratio'],
                                    data_moda_case=self.para_dict['data_moda_case'])
            self.valid_dataset = IXI(root=self.para_dict['data_path'],
                                    modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                    extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                    noise_type='normal',
                                    learn_mode='test',
                                    transform_data=self.normal_transform,
                                    data_mode='paired',
                                    dataset_splited=True)
            self.assigned_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='severe',
                                     learn_mode='test',
                                     transform_data=self.severe_transform,
                                     data_mode='paired',
                                     dataset_splited=False,
                                     assigned_data=self.para_dict['single_img_infer'],
                                     assigned_images=self.para_dict['assigned_images']) 

        self.train_loader = DataLoader(self.train_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=self.para_dict['batch_size'], shuffle=False)
        self.valid_loader = DataLoader(self.valid_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=self.para_dict['batch_size'], shuffle=False)
        self.assigned_loader = DataLoader(self.assigned_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=1, shuffle=False)

        self.client_data_list = self.train_dataset.client_data_indices

    def init_model(self, description='server and clients'):
        pass

    def clients_training(self):
        for i in range(self.para_dict['num_clients']):
            for e in range(self.para_dict['num_epoch']):
                infor = '[Round {}/{}] [Client {}/{}] [Epoch {}/{}] '.format(
                    self.round+1, self.para_dict['num_round'], i+1, self.para_dict['num_clients'], e+1, self.para_dict['num_epoch'])

                self.clients[i].train_epoch(inf=infor)

            # evaluation from a to b
            mae, psnr, ssim, fid = self.clients[i].evaluation(direction='from_a_to_b')
            infor_1 = '{} [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                infor, self.para_dict['source_domain'], self.para_dict['target_domain'], mae, psnr, ssim)
            if self.para_dict['fid']:
                infor_1 = '{} fid: {:.4f}'.format(infor_1, fid)
            print(infor_1)

            if self.para_dict['save_log']:
                save_log(infor_1, self.file_path, description='_clients_from_a_to_b')

            # evaluation from b to a
            mae, psnr_2, ssim, fid = self.clients[i].evaluation(direction='from_b_to_a')
            infor_2 = '{} [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                infor, self.para_dict['target_domain'], self.para_dict['source_domain'], mae, psnr_2, ssim)
            if self.para_dict['fid']:
                infor_2 = '{} fid: {:.4f}'.format(infor_2, fid)
            print(infor_2)

            if self.para_dict['save_log']:
                save_log(infor_2, self.file_path, description='_clients_from_b_to_a')

            # save resuts of psnr for aggregating models 
            self.client_psnr_list.append(psnr)

    def collect_model(self, description='server and clients'):
        pass

    def aggregate_model(self):
        pass

    def transmit_model(self):
        pass

    def save_models(self, psnr):
        if self.para_dict['model'] == 'cyclegan':
            save_model(self.server_gener['from_a_to_b'], '{}/checkpoint/g_from_a_to_b'.format(self.file_path), self.para_dict, psnr)
            save_model(self.server_gener['from_b_to_a'], '{}/checkpoint/g_from_b_to_a'.format(self.file_path), self.para_dict, psnr)
            save_model(self.server_discr['from_a_to_b'], '{}/checkpoint/d_from_a_to_b'.format(self.file_path), self.para_dict, psnr)
            save_model(self.server_discr['from_b_to_a'], '{}/checkpoint/d_from_b_to_a'.format(self.file_path), self.para_dict, psnr)

        elif self.para_dict['model'] == 'munit' or self.para_dict['model'] == 'unit':
            save_model(self.client_gener_list['from_a_to_b_enc'], '{}/checkpoint/g_from_a_to_b_enc'.format(self.file_path), self.para_dict, psnr)
            save_model(self.client_gener_list['from_a_to_b_dec'], '{}/checkpoint/g_from_a_to_b_dec'.format(self.file_path), self.para_dict, psnr)
            save_model(self.client_gener_list['from_b_to_a_enc'], '{}/checkpoint/g_from_b_to_a_enc'.format(self.file_path), self.para_dict, psnr)
            save_model(self.client_gener_list['from_b_to_a_dec'], '{}/checkpoint/g_from_b_to_a_dec'.format(self.file_path), self.para_dict, psnr)
            save_model(self.client_discr_list['from_a_to_b'], '{}/checkpoint/d_from_a_to_b'.format(self.file_path), self.para_dict, psnr)
            save_model(self.client_discr_list['from_b_to_a'], '{}/checkpoint/d_from_a_to_b'.format(self.file_path), self.para_dict, psnr)

    def server_inference(self):
        # evaluation from a to b
        mae, psnr, ssim, fid = self.server.evaluation(direction='from_a_to_b')
        infor = '[Round {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                        self.round+1, self.para_dict['num_round'], self.para_dict['source_domain'], self.para_dict['target_domain'], mae, psnr, ssim)
        if self.para_dict['fid']:
            infor = '{} fid: {:.4f}'.format(infor, fid)
        print(infor)

        if self.para_dict['save_log']:
            save_log(infor, self.file_path, description='_server_from_a_to_b')

        # evaluation from b to a
        mae, psnr, ssim, fid = self.server.evaluation(direction='from_b_to_a')
        infor = '[Round {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
                        self.round+1, self.para_dict['num_round'], self.para_dict['target_domain'], self.para_dict['source_domain'], mae, psnr, ssim)
        if self.para_dict['fid']:
            infor = '{} fid: {:.4f}'.format(infor, fid)
        print(infor)

        if self.para_dict['save_log']:
            save_log(infor, self.file_path, description='_server_from_a_to_b')


        if self.para_dict['plot_distribution']:
            save_img_path = '{}/sample_distribution'.format(self.file_path)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            save_img_path = '{}/round_{}.png'.format(save_img_path, self.round+1)
            self.server.visualize_feature(self.round+1, save_img_path, self.train_loader)
        
        if self.para_dict['save_img']:
            save_img_path = '{}/images/round_{}'.format(self.file_path, self.round+1)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            self.server.infer_images(save_img_path, self.valid_loader)

        if self.para_dict['single_img_infer']:
            save_img_path = '{}/images_assigned/round_{}'.format(self.file_path, self.round+1)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            self.server.infer_images(save_img_path, self.assigned_loader)

        if self.para_dict['save_model']:
            if psnr > self.best_psnr:
                self.save_models(psnr)
                self.best_psnr = psnr

    def collect_feature(self, batch):
        pass

    @torch.no_grad()
    def visualize_feature(self, round, save_img_path, data_loader):
        real_a, fake_a, real_b, fake_b = [], [], [], []
        for i, batch in enumerate(data_loader):
            if i == int(self.config['plot_num_sample'] / self.config['batch_size']):
                break
            real_a_feature, fake_a_feature, real_b_feature, fake_b_feature = self.collect_feature(batch=batch)

            real_a_feature = np.mean(real_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.para_dict['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_a_feature = np.mean(fake_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.para_dict['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            real_b_feature = np.mean(real_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.para_dict['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_b_feature = np.mean(fake_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.para_dict['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))

            if i == 0:
                real_a = real_a_feature
                fake_a = fake_a_feature
                real_b = real_b_feature
                fake_b = fake_b_feature
            else:
                real_a = np.concatenate([real_a, real_a_feature], axis=0)
                fake_a = np.concatenate([fake_a, fake_a_feature], axis=0)
                real_b = np.concatenate([real_b, real_b_feature], axis=0)
                fake_b = np.concatenate([fake_b, fake_b_feature], axis=0)

        plot_sample(real_a, fake_a, real_b, fake_b, step=round, img_path=save_img_path, descript='Round')

        with open(save_img_path.replace('.png', '.npy'), 'wb') as f:
            np.save(f, np.array([real_a, fake_a, real_b, fake_b]))

    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.load_data()
        self.init_model()

        for round in range(self.para_dict['num_round']):
            self.round = round
            # train clients model 
            self.clients_training()
            # collect models of clients when client model training is finished
            self.collect_model()
            # update server model
            self.aggregate_model()
            # test server performance and infer images for viualization
            self.server_inference()
            # update client model
            self.transmit_model() 

        print('work dir {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f) 
        print('---------------------')