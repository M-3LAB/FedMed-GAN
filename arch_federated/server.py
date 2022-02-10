from collections import OrderedDict 

__all__ = ['aggregate_from_client_to_server', 'update_server_from_best_psnr_client']

def aggregate_from_client_to_server(server, clients, aggregation_weights):
    update_state = OrderedDict() 
    for k, client in enumerate(clients):
        local_state = client.state_dict() 
        for key in server.state_dict().keys():
            if k == 0:
                update_state[key] = local_state[key] * aggregation_weights[k] 
            else:
                update_state[key] += local_state[key] * aggregation_weights[k] 

    server.load_state_dict(update_state)

def update_server_from_best_psnr_client(server, clients, psnr_list):
    max_psnr = max(psnr_list)
    max_psnr_index = psnr_list.index(max_psnr)
    best_psnr_client = clients[max_psnr_index] 

    update_state = OrderedDict() 
    local_state = best_psnr_client.state_dict()

    for key in server.state_dict().keys():
        update_state[key] = local_state[key]

    server.load_state_dict(update_state)    

