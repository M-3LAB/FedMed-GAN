from arch_federated.federated_learning import FederatedTrain
from arch_centralized.cyclegan import CycleGAN
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from arch_federated.server import aggregate_from_client_to_server, update_server_from_best_psnr_client
from arch_federated.client import send_from_server_to_client

__all__ = ['FedCycleGAN']
class FedCycleGAN(FederatedTrain):
    def __init__(self, args):
        super(FedCycleGAN, self).__init__(args=args)
        self.args = args

    def init_model(self, description='server and clients'):
        self.server = CycleGAN(self.para_dict, self.train_loader, self.valid_loader,
                               self.assigned_loader, self.device, self.file_path)
        self.clients = []
        for i in range(self.para_dict['num_clients']):
            client_train_loader = DataLoader(self.train_dataset,
                                             batch_size=self.para_dict['batch_size'],
                                             drop_last=True,
                                             num_workers=self.para_dict['num_workers'],
                                             sampler=SubsetRandomSampler(self.client_data_list[i]))
            self.clients.append(CycleGAN(self.para_dict, client_train_loader, self.valid_loader,
                                self.assigned_loader, self.device, self.file_path, self.para_dict['clients_data_weight'][i]))

    def collect_model(self, description='server and clients'):
        self.server_gener['from_a_to_b'] = self.server.generator_from_a_to_b
        self.server_gener['from_b_to_a'] = self.server.generator_from_b_to_a

        self.server_discr['from_a_to_b'] = self.server.discriminator_from_a_to_b
        self.server_discr['from_b_to_a'] = self.server.discriminator_from_b_to_a

        # clear old weights
        self.client_gener_list = {'from_a_to_b': [], 'from_b_to_a': []} 
        for i in range(self.para_dict['num_clients']):
            self.client_gener_list['from_a_to_b'].append(self.clients[i].generator_from_a_to_b)
            self.client_gener_list['from_b_to_a'].append(self.clients[i].generator_from_b_to_a)

            self.client_discr_list['from_a_to_b'].append(self.clients[i].discriminator_from_a_to_b)
            self.client_discr_list['from_b_to_a'].append(self.clients[i].discriminator_from_b_to_a)

    def aggregate_model(self):
        if self.para_dict['fed_aggregate_method'] == 'fed-avg':
            aggregate_from_client_to_server(self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'],
                      aggregation_weights=self.para_dict['clients_data_weight'])
            aggregate_from_client_to_server(self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'],
                      aggregation_weights=self.para_dict['clients_data_weight'])

        elif self.para_dict['fed_aggregate_method'] == 'fed-psnr':
            update_server_from_best_psnr_client(
                self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'], self.client_psnr_list)
            update_server_from_best_psnr_client(
                self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'], self.client_psnr_list)
            # clear
            self.client_psnr_list = []
        else:
            raise ValueError

    def transmit_model(self):
        send_from_server_to_client(
            self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'])
        send_from_server_to_client(
            self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'])

    def collect_feature(self, batch):
        real_a = batch[self.para_dict['source_domain']].to(self.device)
        real_b = batch[self.para_dict['target_domain']].to(self.device)

        fake_a = self.server.generator_from_b_to_a(real_b)
        fake_b = self.server.generator_from_a_to_b(real_a)

        real_a_feature = self.server.generator_from_a_to_b.extract_feature(real_a)
        fake_a_feature = self.server.generator_from_a_to_b.extract_feature(fake_a)
        real_b_feature = self.server.generator_from_b_to_a.extract_feature(real_b)
        fake_b_feature = self.server.generator_from_b_to_a.extract_feature(fake_b)

        return real_a_feature, fake_a_feature, real_b_feature, fake_b_feature



    def run_work_flow(self):
        return super().run_work_flow()