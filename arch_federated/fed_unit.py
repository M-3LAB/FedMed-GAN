from arch_federated.federated_learning import FederatedTrain
from arch_centralized.unit import Unit
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from arch_federated.server import aggregate_from_client_to_server, update_server_from_best_psnr_client
from arch_federated.client import send_from_server_to_client

class FedUnit(FederatedTrain):
    def __init__(self, args):
        super(FedUnit, self).__init__(args=args)
        self.args = args


    def init_model(self, description='server and clients'):
        self.server = Unit(self.para_dict, self.train_loader, self.valid_loader,
                               self.assigned_loader, self.device, self.file_path)
        self.clients = []
        for i in range(self.para_dict['num_clients']):
            client_train_loader = DataLoader(self.train_dataset,
                                             batch_size=self.para_dict['batch_size'],
                                             drop_last=True,
                                             num_workers=self.para_dict['num_workers'],
                                             sampler=SubsetRandomSampler(self.client_data_list[i]))
            self.clients.append(Unit(self.para_dict, client_train_loader, self.valid_loader,
                                self.assigned_loader, self.device, self.file_path, self.para_dict['clients_data_weight'][i]))

    def collect_model(self, description='server and clients'):
        self.server_gener['from_a_to_b_enc'] = self.server.generator_from_a_to_b_enc
        self.server_gener['from_a_to_b_dec'] = self.server.generator_from_a_to_b_dec
        self.server_gener['from_b_to_a_enc'] = self.server.generator_from_b_to_a_enc
        self.server_gener['from_b_to_a_dec'] = self.server.generator_from_b_to_a_dec

        self.server_discr['from_a_to_b'] = self.server.discriminator_from_a_to_b
        self.server_discr['from_b_to_a'] = self.server.discriminator_from_b_to_a

        # clear old weights
        self.client_gener_list = {'from_a_to_b_enc': [], 'from_a_to_b_dec': [], 'from_b_to_a_enc': [], 'from_b_to_a_dec': []} 
        for i in range(self.para_dict['num_clients']):
            self.client_gener_list['from_a_to_b_enc'].append(self.clients[i].generator_from_a_to_b_enc)
            self.client_gener_list['from_a_to_b_dec'].append( self.clients[i].generator_from_a_to_b_dec)
            self.client_gener_list['from_b_to_a_enc'].append(self.clients[i].generator_from_b_to_a_enc)
            self.client_gener_list['from_b_to_a_dec'].append(self.clients[i].generator_from_b_to_a_dec)

            self.client_discr_list['from_a_to_b'].append(self.clients[i].discriminator_from_a_to_b)
            self.client_discr_list['from_b_to_a'].append(self.clients[i].discriminator_from_b_to_a)

    def aggregate_model(self):
        if self.para_dict['fed_aggregate_method'] == 'fed-avg':
            aggregate_from_client_to_server(self.server_gener['from_a_to_b_enc'], self.client_gener_list['from_a_to_b_enc'],
                      aggregation_weights=self.para_dict['clients_data_weight'])
            aggregate_from_client_to_server(self.server_gener['from_a_to_b_dec'], self.client_gener_list['from_a_to_b_dec'],
                      aggregation_weights=self.para_dict['clients_data_weight'])

            aggregate_from_client_to_server(self.server_gener['from_b_to_a_enc'], self.client_gener_list['from_b_to_a_enc'],
                      aggregation_weights=self.para_dict['clients_data_weight'])
            aggregate_from_client_to_server(self.server_gener['from_b_to_a_dec'], self.client_gener_list['from_b_to_a_dec'],
                      aggregation_weights=self.para_dict['clients_data_weight'])

        elif self.para_dict['fed_aggregate_method'] == 'fed-psnr':
            update_server_from_best_psnr_client(
                self.server_gener['from_a_to_b_enc'], self.client_gener_list['from_a_to_b_enc'], self.client_psnr_list)
            update_server_from_best_psnr_client(
                self.server_gener['from_a_to_b_dec'], self.client_gener_list['from_a_to_b_dec'], self.client_psnr_list)
            update_server_from_best_psnr_client(
                self.server_gener['from_b_to_a_enc'], self.client_gener_list['from_b_to_a_enc'], self.client_psnr_list)
            update_server_from_best_psnr_client(
                self.server_gener['from_b_to_a_dec'], self.client_gener_list['from_b_to_a_dec'], self.client_psnr_list)
            # clear
            self.client_psnr_list = []

        else:
            raise ValueError

    def transmit_model(self):
        send_from_server_to_client(self.server_gener['from_a_to_b_enc'], self.client_gener_list['from_a_to_b_enc'])
        send_from_server_to_client(self.server_gener['from_a_to_b_dec'], self.client_gener_list['from_a_to_b_dec'])

        send_from_server_to_client(self.server_gener['from_b_to_a_enc'], self.client_gener_list['from_b_to_a_enc'])
        send_from_server_to_client(self.server_gener['from_b_to_a_dec'], self.client_gener_list['from_b_to_a_dec'])

    def run_work_flow(self):
        return super().run_work_flow()