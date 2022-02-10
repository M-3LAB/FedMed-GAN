from collections import OrderedDict 

__all__ = ['send_from_server_to_client']


def send_from_server_to_client(server, clients):
    server_state = server.state_dict()
    for k, client in enumerate(clients):
        local_state = client.state_dict()
        for key in server.state_dict().keys():
            local_state[key] = server_state[key]

        client.load_state_dict(local_state)