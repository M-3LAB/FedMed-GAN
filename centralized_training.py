from configuration.config import parse_arguments_centralized
from arch_centralized.centralized_learning import CentralizedTrain


def centralized_training():
    args = parse_arguments_centralized()
    
    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} must be refered!'.format(key_arg))

    work = CentralizedTrain(args=args)
    work.run_work_flow()



if __name__ == '__main__':
    centralized_training()