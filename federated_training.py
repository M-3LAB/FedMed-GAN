from configuration.config import parse_arguments_federated
from arch_federated.fed_cyclegan import Fed_CycleGAN
from arch_federated.fed_munit import Fed_Munit
from arch_federated.fed_unit import Fed_Unit



def federated_training():
    args = parse_arguments_federated()

    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} must be refered!'.format(key_arg))

    if args.model == 'cyclegan':
        work = Fed_CycleGAN(args=args)
    elif args.model == 'munit':
        work = Fed_Munit(args=args)
    elif args.model == 'unit':
        work = Fed_Unit(args=args)
    else:
        raise ValueError('Model is invalid!')   

    work.run_work_flow()





if __name__ == '__main__':
    federated_training()