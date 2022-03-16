import torch
from configuration.config import parse_arguments_nirps
import yaml
import os
from tools.utilize import *
from data_io.brats import BraTS2021
from data_io.ixi import IXI
from torch.utils.data import DataLoader
from arch_centralized.cyclegan import CycleGAN
from arch_centralized.munit import Munit
from arch_centralized.unit import Unit

import warnings
warnings.filterwarnings("ignore")

class NIRPS(object):
    def __init__(self, args):
        self.args = args

    def load_config(self):
        with open('./configuration/nirps/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/nirps/2_train_base/centralized_training.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/nirps/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)

        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)

    def preliminary(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

        seed_everything(self.para_dict['seed'])

        device, device_ids = parse_device_list(self.para_dict['gpu_ids'], int(self.para_dict['gpu_id']))
        self.device = torch.device("cuda", device)

        self.file_path = record_path(self.para_dict)
        #save log
        save_arg(self.para_dict, self.file_path)
        save_script(__file__, self.file_path)

        self.fid_stats_from_a_to_b = '{}/{}/{}_{}_fid_stats.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['source_domain'], self.para_dict['target_domain'])
        self.fid_stats_from_b_to_a = '{}/{}/{}_{}_fid_stats.npz'.format(
            self.para_dict['fid_dir'], self.para_dict['dataset'], self.para_dict['target_domain'], self.para_dict['source_domain'])

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

    def setup_folder(self):
        # fp: file path
        self.dataset_fp = self.para_dict['dataset']
        create_folders(self.dataset_fp)

        self.source_modality_fp = os.path.join(self.dataset_fp, self.para_dict['source_domain'])
        create_folders(self.source_modality_fp)

        self.target_modality_fp = os.path.join(self.dataset_fp, self.para_dict['target_domain'])
        create_folders(self.target_modality_fp)

        self.model_source_fp = os.path.join(self.source_modality_fp, self.para_dict['model']) 
        self.model_target_fp = os.path.join(self.target_modality_fp, self.para_dict['model'])
        create_folders(self.model_source_fp)
        create_folders(self.model_target_fp)

        self.model_source_gt_fp = os.path.join(self.model_source_fp, 'gt')
        self.model_target_gt_fp = os.path.join(self.model_target_fp, 'gt')

        for i in range(self.para_dict['num_epoch']):
            epoch_model_source_fp = os.path.join(self.model_source_fp, str(i)) 
            epoch_model_target_fp = os.path.join(self.model_target_fp, str(i)) 
            create_folders(epoch_model_source_fp)
            create_folders(epoch_model_target_fp)
            

    def load_data(self):
        self.normal_transform = [{'degrees':0, 'translate':[0.00, 0.00],
                                     'scale':[1.00, 1.00], 
                                     'size':(self.para_dict['size'], self.para_dict['size'])},
                                 {'degrees':0, 'translate':[0.00, 0.00],
                                  'scale':[1.00, 1.00], 
                                  'size':(self.para_dict['size'], self.para_dict['size'])}]

        if self.para_dict['dataset'] == 'brats2021':
            assert self.para_dict['source_domain'] in ['t1', 't2', 'flair']
            assert self.para_dict['target_domain'] in ['t1', 't2', 'flair']

            self.train_dataset = BraTS2021(root=self.para_dict['data_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           noise_type='normal',
                                           learn_mode='train',
                                           transform_data=self.normal_transform,
                                           client_weights=[1.0],
                                           data_mode=self.para_dict['data_mode'],
                                           data_num=self.para_dict['data_num'],
                                           data_paired_weight=1.0,
                                           data_moda_ratio=1.0,
                                           data_moda_case='case1')

            self.valid_dataset = BraTS2021(root=self.para_dict['valid_path'],
                                           modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                           noise_type='normal',
                                           learn_mode='test',
                                           extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                           transform_data=self.normal_transform,
                                           data_mode='paired',
                                           assigned_data=True,
                                           assigned_images=None) 
            
        elif self.para_dict['dataset'] == 'ixi':
            assert self.para_dict['source_domain'] in ['t2', 'pd']
            assert self.para_dict['target_domain'] in ['t2', 'pd']

            self.train_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='train',
                                     transform_data=self.normal_transform,
                                     data_mode=self.para_dict['data_mode'],
                                     data_num=self.para_dict['data_num'],
                                     data_paired_weight=1.0,
                                     client_weights=[1.0],
                                     dataset_splited=True,
                                     data_moda_ratio=1.0,
                                     data_moda_case='case1')

            self.valid_dataset = IXI(root=self.para_dict['data_path'],
                                     modalities=[self.para_dict['source_domain'], self.para_dict['target_domain']],
                                     extract_slice=[self.para_dict['es_lower_limit'], self.para_dict['es_higher_limit']],
                                     noise_type='normal',
                                     learn_mode='test',
                                     transform_data=self.normal_transform,
                                     data_mode='paired',
                                     dataset_splited=True) 
        else:
            raise NotImplementedError('This Dataset Has Not Been Implemented Yet')

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.para_dict['batch_size'],
                                       num_workers=self.para_dict['num_workers'],
                                       shuffle=True)

        self.valid_loader = DataLoader(self.valid_dataset, 
                                       num_workers=self.para_dict['num_workers'],
                                       batch_size=1, 
                                       shuffle=False)
        
        self.assigned_loader = None

    def init_model(self):
        if self.para_dict['model'] == 'cyclegan':
            self.trainer = CycleGAN(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        elif self.para_dict['model'] == 'munit':
            self.trainer = Munit(self.para_dict, self.train_loader, self.valid_loader,
                                    self.assigned_loader, self.device, self.file_path)
        elif self.para_dict['model'] == 'unit':
            self.trainer = Unit(self.para_dict, self.train_loader, self.valid_loader,
                                self.assigned_loader, self.device, self.file_path)
        else:
            raise ValueError('Model is invalid!')

    def save_models(self, fp=None, epoch=None):
        if self.para_dict['model'] == 'cyclegan':
            gener_from_a_to_b, gener_from_b_to_a, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model_per_epoch(gener_from_a_to_b, '{}/checkpoint/g_from_a_to_b'.format(fp), self.para_dict, epoch)
            save_model_per_epoch(gener_from_b_to_a, '{}/checkpoint/g_from_b_to_a'.format(fp), self.para_dict, epoch)
           
        elif self.para_dict['model'] == 'munit' or self.para_dict['model'] == 'unit':
            gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc, gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a = self.trainer.get_model()
            save_model(gener_from_a_to_b_enc, '{}/checkpoint/g_from_a_to_b_enc'.format(fp), self.para_dict, epoch)
            save_model(gener_from_a_to_b_dec, '{}/checkpoint/g_from_a_to_b_dec'.format(fp), self.para_dict, epoch)
            save_model(gener_from_b_to_a_enc, '{}/checkpoint/g_from_b_to_a_enc'.format(fp), self.para_dict, epoch)
            save_model(gener_from_b_to_a_dec, '{}/checkpoint/g_from_b_to_a_dec'.format(fp), self.para_dict, epoch)

    def work_flow(self):
        self.trainer.train_epoch()
        # evaluation from a to b
        if self.para_dict['general_evaluation']:
            (source_mae, source_psnr, source_ssim, source_fid, target_mae, target_psnr, target_ssim, target_fid) = self.trainer.evaluation(direction='both')

            src_infor = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['source_domain'], source_mae, source_psnr, source_ssim, source_fid)

            tag_infor = '[Epoch {}/{}] [{}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f} fid: {:.4f}'.format(
                self.epoch+1, self.para_dict['num_epoch'], self.para_dict['target_domain'], target_mae, target_psnr, target_ssim, target_fid)
        
            print(src_infor)
            print(tag_infor)
        
            save_log(src_infor, self.file_path, description='both')
            save_log(tag_infor, self.file_path, description='both')
        
        for i in range(int(self.para_dict['num_epoch'])):
            epoch_model_source_fp = os.path.join(self.model_source_fp, str(i)) 
            epoch_model_target_fp = os.path.join(self.model_target_fp, str(i)) 
            self.save_models(fp=epoch_model_source_fp, epoch=i)
            self.save_models(fp=epoch_model_target_fp, epoch=i)
            self.trainer.infer_nirps_generated(src_epoch_path=epoch_model_source_fp,
                                               tag_epoch_path=epoch_model_target_fp,
                                               data_loader=self.valid_loader)
        
        self.trainer.infer_nirps_gt(src_gt_path=self.model_source_gt_fp,
                                    tag_gt_path=self.model_target_gt_fp,
                                    data_loader=self.valid_loader)
            
    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.setup_folder()
        self.load_data()
        self.init_model()
        print('---------------------')

        for epoch in range(self.para_dict['num_epoch']):
            self.epoch = epoch
            self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

if __name__ == '__main__':
    args = parse_arguments_nirps()

    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} must be refered!'.format(key_arg))

    work = NIRPS(args=args)
    work.run_work_flow()