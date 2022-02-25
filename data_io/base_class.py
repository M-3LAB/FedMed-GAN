from scipy import rand
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from data_io.noise import GaussianNoise

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-8)
        return torch.from_numpy(tensor)

class BASE_DATASET(torch.utils.data.Dataset):
    """Dataset utility class.

    Args:
        root: (str) Path of the folder with all the images.
        mode : {'train' or 'test'} Part of the dataset that is loaded.
        extract_slice: [start, end] Extract slice of one volume id
        data_mode: mixed, which if True, real-world data setting, which blends paired data and unpaired data
        clients: (list) Client weights when splitting the whole data
        splited: If True, we want to split the data into two parts, i.e, training data(0.8) and testing data(0.2)

    """
    def __init__(self, root, modalities=["t1", "t2"], learn_mode="train", extract_slice=[29, 100],
                noise_type='normal', transform_data=None, client_weights=[1.0], dataset_splited=False,
                data_mode='mixed', data_num=6000, data_paired_weight=0.2, data_moda_ratio=0.5, data_moda_case='case1'):
        random.seed(3)
        
        self.dataset_path = root
        self.extract_slice = extract_slice
        self.client_weights = client_weights
        # data setting
        self.learn_mode = learn_mode
        self.dataset_splited = dataset_splited
        self.data_mode = data_mode
        self.data_num = data_num
        self.data_paired_weight = data_paired_weight
        self.data_moda_ratio = data_moda_ratio
        self.data_moda_case = data_moda_case
        # data augumentaion
        self.noise_type = noise_type
        self.t= transform_data
        self.modality_a = modalities[0]
        self.modality_b = modalities[1]
        self.transform_a = None
        self.transform_b = None
        # data generation
        self.files = []  # volume name of whole dataset
        self.train_files = []  # volume id in trainset
        self.valid_files = []  # volume id in validset
        self.all_data = []  # slice id of all cases, including paired, unpaired, mixed
        self.client_data = [] # all client indices
        self.client_indice_container = [] # all cases with file name
        self.data_total_num_list = [] # record num by [paired, unpaired]
        # dataloader used
        self.fedmed_dataset = []  # slice id of cases for training
        self.client_data_indices = [] # all client indices for training

    def __getitem__(self, index):
        path_a, path_b, i = self.fedmed_dataset[index]
        moda_a = np.load('{}/{}/{}.npy'.format(self.dataset_path, self.modality_a.upper(), path_a))
        moda_b = np.load('{}/{}/{}.npy'.format(self.dataset_path, self.modality_b.upper(), path_b))
        
        if len(moda_a.shape) == 2 and len(moda_b.shape) == 2:
            moda_a = moda_a[:, :]
            moda_b = moda_b[:, :]
        elif len(moda_a.shape) == 3 and len(moda_b.shape) == 3:
            moda_a = moda_a[i, :, :]
            moda_b = moda_b[i, :, :]
        else:
            raise ValueError('load file failed!')

        data_a = self.transform_a(moda_a.astype(np.float32))
        data_b = self.transform_b(moda_b.astype(np.float32))

        # check transformed results
        # plt.subplot(121)
        # plt.imshow(moda_a, cmap='gray') 
        # plt.title('input')
        # plt.subplot(122)
        # plt.title('transformed')
        # plt.imshow(data_a.squeeze(), cmap='gray') 
        # plt.savefig('./legacy_code/img_after_{}.jpg'.format(i))

        return {self.modality_a: data_a, self.modality_b: data_b, 
                'name_a': path_a, 'name_b': path_b, 'slice_num': i}

    def __len__(self):
        return len(self.fedmed_dataset)

    def _check_sanity(self):
        """
        obtain file names, which are saved into self.files
        """
        pass

    def _check_noise_type(self):
        """
        noise type check, i.e., normal, gaussian and reg
        """
        if self.noise_type == 'normal':
            assert 'size' in list(self.t[0].keys()) 
        elif self.noise_type == 'slight':
            assert 'degrees' in list(self.t[0].keys()) 
            assert 'translate' in list(self.t[0].keys()) 
            assert 'scale' in list(self.t[0].keys()) 
            assert 'size' in list(self.t[0].keys()) 
        elif self.noise_type == 'gaussian':
            assert 'mu' in list(self.t[0].keys()) 
            assert 'sigma' in list(self.t[0].keys()) 
            assert 'size' in list(self.t[0].keys()) 
        elif self.noise_type == 'severe':
            assert 'degrees' in list(self.t[0].keys()) 
            assert 'translate' in list(self.t[0].keys()) 
            assert 'scale' in list(self.t[0].keys()) 
            assert 'size' in list(self.t[0].keys()) 
        else:
            raise ValueError('Noise Hyperparameter Setting Incorrect')

    def _get_transform_modalities(self):
        """
        obtain transform, which are saved into self.transform_modalities
        """
        if self.noise_type == 'normal':
            self.transform_a = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.Resize(size=self.t[0]['size']),
                                                   ToTensor()])
            self.transform_b = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.Resize(size=self.t[1]['size']), 
                                                   ToTensor()])
        elif self.noise_type == 'slight':
            self.transform_a = transforms.Compose([transforms.ToPILImage(), 
                                                    transforms.RandomAffine(degrees=self.t[0]['degrees'], translate=self.t[0]['translate'], 
                                                                            scale=self.t[0]['scale'], fillcolor=0), 
                                                    transforms.Resize(size=self.t[0]['size']),
                                                    ToTensor()])
            self.transform_b = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.RandomAffine(degrees=self.t[1]['degrees'], translate=self.t[1]['translate'], 
                                                                            scale=self.t[1]['scale'], fillcolor=0), 
                                                   transforms.Resize(size=self.t[1]['size']), 
                                                   ToTensor()])
        elif self.noise_type == 'gaussian':
            self.transform_a = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.Resize(size=self.t[0]['size']),
                                                   ToTensor(),
                                                   GaussianNoise(mean=self.t[0]['mu'],
                                                                 std=self.t[0]['sigma'])])
            self.transform_b = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.Resize(size=self.t[1]['size']), 
                                                   ToTensor(),
                                                   GaussianNoise(mean=self.t[1]['mu'],
                                                                 std=self.t[1]['sigma'])])
        elif self.noise_type == 'severe':
            self.transform_a = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.RandomAffine(degrees=self.t[0]['degrees'], translate=self.t[0]['translate'], 
                                                                            scale=self.t[0]['scale'], fillcolor=0), 
                                                   transforms.Resize(size=self.t[0]['size']),
                                                   ToTensor()])
            self.transform_b = transforms.Compose([transforms.ToPILImage(), 
                                                   transforms.RandomAffine(degrees=self.t[1]['degrees'], translate=self.t[1]['translate'], 
                                                                            scale=self.t[1]['scale'], fillcolor=0), 
                                                   transforms.Resize(size=self.t[1]['size']), 
                                                   ToTensor()])
        else:
            raise ValueError('Noise Type Setting Incorrect')
                                                   
    def _generate_dataset(self):

        file_container = None
        if self.dataset_splited:
            # grab volumes, which are devided into trainset and validset
            dataset_indice = self._allocate_client_data(data_len=len(self.files), clients=[0.8, 0.2])
            self.train_files = [self.files[i] for i in dataset_indice[0]]
            self.valid_files = [self.files[i] for i in dataset_indice[1]]
        else:
            self.train_files = self.files
            self.valid_files = self.files

        if self.learn_mode == 'train':
            file_container = self.train_files
        elif self.learn_mode == 'test':
            file_container = self.valid_files
            self.client_weights = [1.0]
        else:
            raise NotImplementedError('Train Mode is Wrong')

        # seperated volume ids into clients
        file_indices = self._allocate_client_data(data_len=len(file_container), clients=self.client_weights)

        for client_indices in file_indices:
            paired, unpaired = [], []
            # grab volumes into each client
            client_files = [file_container[i] for i in client_indices]

            # get paired data indices
            for i in range(len(client_files)):
                for j in range(self.extract_slice[0], self.extract_slice[1]):
                    index_para = [client_files[i], client_files[i], j]
                    paired.append(index_para)
            self.client_indice_container.append(paired)

            # get unpaired data indices 
            moda_b_indices = random.sample(client_indices, int(len(client_indices) *(1. - self.data_moda_ratio)))
            moda_a_indices = list(set(client_indices) - set(moda_b_indices))
            moda_a_files = [file_container[i] for i in moda_a_indices]
            moda_b_files = [file_container[i] for i in moda_b_indices]

            # case 1, moda_a in moda_a_files, moda_b in moda_b_files
            if self.data_moda_case == 'case1':
                pass
            # case 2, moda_a in moda_a_files, moda_b in client_flies(all)
            if self.data_moda_case == 'case2':
                moda_b_files = client_files

            for i in range(len(moda_a_files)):
                for j in range(len(moda_b_files)):
                    for k in range(self.extract_slice[0], self.extract_slice[1]):
                        index_para = [moda_a_files[i], moda_b_files[j], k]
                        unpaired.append(index_para)
            self.client_indice_container.append(unpaired)

        # generate one list, [[moda A name, moda B name, i-th slice], ...]
        self.all_data = [x for inner_list in self.client_indice_container for x in inner_list]


    def _generate_client_indice(self):
        dataset_indices = [i for i in range(len(self.all_data))]
        client_data_list = []
        mixed_data_num_list = []
        start = 0

        # get the indices of each client data in all_data
        for client in self.client_indice_container:
            mixed_data_num_list.append(len(client))
            end = start + len(client)
            indice = dataset_indices[start:end]
            client_data_list.append(indice)
            start = end

        # shuffle each client data indices
        for i in range(len(self.client_weights)):
            paired_data = client_data_list[i*2]
            unpaired_data = client_data_list[i*2+1]
            random.shuffle(paired_data)
            random.shuffle(unpaired_data)

            self.client_data.append([paired_data, unpaired_data])
            self.data_total_num_list.append([mixed_data_num_list[i*2], mixed_data_num_list[i*2+1]])

        # get the desired number of indices
        for i in range(len(self.client_weights)):
            data_num = int(self.data_num * self.client_weights[i])

            if self.data_mode == 'mixed':
                paired_num = int(data_num * self.data_paired_weight)
                unpaired_num = data_num - paired_num

                if paired_num > self.data_total_num_list[i][0] or unpaired_num > self.data_total_num_list[i][1]:
                    raise ValueError('Not Enough Desired Data')

                paired_data = self.client_data[i][0][:paired_num]
                unpaired_data = self.client_data[i][1][:unpaired_num]
                mixed_data = paired_data + unpaired_data
                random.shuffle(mixed_data)
                self.client_data_indices.append(mixed_data)

            elif self.data_mode == 'paired':
                paired_data = []
                if self.learn_mode == 'train':
                    paired_data = self.client_data[i][0][:data_num]
                else:
                    paired_data = self.client_data[i][0][:] # all cases
                random.shuffle(paired_data)
                self.client_data_indices.append(paired_data)

            elif self.data_mode == 'unpaired':
                unpaired_data = []
                if self.learn_mode == 'train':
                    unpaired_data = self.client_data[i][1][:data_num]
                else:
                    unpaired_data = self.client_data[i][1][:6000] # manually set
                random.shuffle(unpaired_data)
                self.client_data_indices.append(unpaired_data)
               
            else:
                raise NotImplementedError('Data Mode is Wrong')
            
            self.fedmed_dataset = self.all_data

    @staticmethod
    def _allocate_client_data(data_len, clients=[1.0]):
        dataset_indices = [i for i in range(data_len)]
        random.shuffle(dataset_indices)
        
        start = 0
        client_data_inidces=[]
        for ratio in clients:
            end = start + round(ratio * data_len)
            if end > data_len:
                end = data_len
            indice = dataset_indices[start:end]
            client_data_inidces.append(indice)
            start = end

        return client_data_inidces
