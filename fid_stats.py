import torch
import os
import yaml

from metrics.fid_is.fid import get_stats
from data_io.brats import BraTS2021 
from data_io.ixi import IXI
from tools.utilize import *
from torch.utils.data import DataLoader
from configuration.config import parse_arguments_fid_stats


def fid_stats(args):
    with open('./configuration/fid_stats/fid_stats_{}.yaml'.format(args.dataset), 'r') as f:
       para_dict = yaml.load(f, Loader=yaml.SafeLoader)
    para_dict = merge_config(para_dict, args)
    print(para_dict)

    output_path = os.path.join(para_dict['fid_dir'], para_dict['dataset'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    seed_everything(para_dict['seed'])

    device, device_ids = parse_device_list(para_dict['gpu_ids'], int(para_dict['gpu_id']))
    device = torch.device("cuda", device)

    #get_stats()
    test_transform_data = [{'degrees':0, 'translate':[0.00, 0.00],
                            'scale':[1.00, 1.00], 'size':(para_dict['size'], para_dict['size']),
                            'fillcolor':0},
                           {'degrees':0, 'translate':[0.00, 0.00],
                            'scale':[1.00, 1.00], 'size':(para_dict['size'], para_dict['size']),
                            'fillcolor':0}]
    
    #Dataset IO
    if para_dict['dataset'] == 'ixi':
        assert para_dict['source_domain'] in ['t2', 'pd']
        assert para_dict['target_domain'] in ['t2', 'pd']

        test_dataset = IXI(root=para_dict['data_path'],
                           modalities=[para_dict['source_domain'], para_dict['target_domain']],
                           extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                           noise_type='normal',
                           learn_mode='test',
                           transform_data=test_transform_data,
                           data_mode='paired',
                           dataset_splited=True)

    elif para_dict['dataset'] == 'brats2021':
        assert para_dict['source_domain'] in ['t1', 't2', 'flair']
        assert para_dict['target_domain'] in ['t1', 't2', 'flair']

        test_dataset = BraTS2021(root=para_dict['valid_path'],
                                 modalities=[para_dict['source_domain'], para_dict['target_domain']],
                                 learn_mode='test',
                                 extract_slice=[para_dict['es_lower_limit'], para_dict['es_higher_limit']],
                                 noise_type='normal',
                                 transform_data=test_transform_data,
                                 data_mode='paired')
    else:
        raise NotImplementedError('Dataset is Invalid!')

    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=para_dict['batch_size'], shuffle=False)


    get_stats(test_loader, para_dict['batch_size'], output_path,
              para_dict['source_domain'], para_dict['target_domain'], device)


if __name__ == '__main__':
    args = parse_arguments_fid_stats()
    fid_stats(args)