import argparse

__all__ = ['parse_argument_bise', 'parse_arguments_federated', 'parse_arguments_nirps', 
            'parse_arguments_centralized', 'parse_arguments_fid_stats']


def parse_arguments_nirps():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--dataset', '-d', type=str, default='ixi', choices=['ixi', 'brats2021'])
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--general-evaluation', action='store_true', default=None, help='indicate whether the evaluation for total images need to be done or not')
    parser.add_argument('--source-domain', '-s', default='pd', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)
    parser.add_argument('--data-mode', '-dm', type=str, default='mixed', choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-num', type=int, default=None, help='slices number for GAN training')
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--atl', action='store_true', default=None, help='indicate whether the atl flag is true or not')
    parser.add_argument('--debug', action='store_true', default=None)
    #parser.add_argument('--nirps-structure', '-ns', action='store_true', help='flag for nirps structure')
    args = parser.parse_args()
    return args

def parse_arguments_kaid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['normal', 'gaussian', 'slight', 'severe'])
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--msl-stats', action='store_true', help='mask stastical learning')
    parser.add_argument('--msl-assigned', action='store_true', help='mask assigned flag')
    parser.add_argument('--msl-assigned-value', type=float, help='msl assigned value')
    parser.add_argument('--msl-path', type=str, default=None, help='mask side length storage path')
    parser.add_argument('--delta-diff', type=float, default=None, help='mask side length difference vairation thereshold value')
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--lambda-recon', type=float, default=1.0, help='weight for reconstruction loss')
    parser.add_argument('--lambda-contrastive', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--lambda-hf', type=float, default=1.0, help='weight for high frequency part')
    parser.add_argument('--lambda-lf', type=float, default=1.0, help='weight for low frequency part')
    parser.add_argument('--pair-num', '-pn', type=int, default=10000)
    parser.add_argument('--test-model', type=str, default='cyclegan', choices=['cyclegan','munit','unit'])
    #parser.add_argument('--mode', type=str, default='train', choices=['train', 'pred', 'trainpred'])
    parser.add_argument('--diff-method', type=str, default=None, choices=['l1', 'l2', 'cos'])
    parser.add_argument('--source-domain', '-s', type=str, default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', type=str, default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--step-size', type=int, default=None, help='learning rate will be adjust for epoch numbers')
    parser.add_argument('--gamma', type=float, default=None, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--beta1', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--beta2', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--fid', action='store_true', default=True)
    args = parser.parse_args()
    return args


def parse_arguments_federated():
    parser = argparse.ArgumentParser()
    # federated setting
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default=None)
    parser.add_argument('--num-round', type=int, default=10)
    parser.add_argument('--num-clients', type=int, default=None)
    parser.add_argument('--clients-data-weight', type=float, default=None, nargs='+')
    parser.add_argument('--clip-bound', type=float, default=None)
    parser.add_argument('--noise-multiplier', type=float, default=None)
    parser.add_argument('--not-test-client', '-ntc', action='store_true', default=False)

    # centralized setting
    parser.add_argument('--dataset', '-d', type=str, default='ixi', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', default='pd', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default='mixed', choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=0.5, choices=[0., 0.1, 0.3, 0.5, 1.])

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=3)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--batch-size', type=int, default=None)

    parser.add_argument('--diff-privacy', action='store_true', default=None) 
    parser.add_argument('--identity', action='store_true', default=False)
    parser.add_argument('--reg-gan', action='store_true', default=False)
    parser.add_argument('--fid', action='store_true', default=True)

    parser.add_argument('--auxiliary-rotation', '-ar', action='store_true', default=False)
    parser.add_argument('--auxiliary-translation', '-at', action='store_true', default=False)
    parser.add_argument('--auxiliary-scaling', '-as', action='store_true', default=False)

    parser.add_argument('--noise-level', '-nl', type=int, default=None, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--noise-type', '-nt', type=str, default=None, choices=['normal', 'slight', 'severe'])

    # input noise augmentation method
    parser.add_argument('--severe-rotation', '-sr', type=float, default=None, choices=[15, 30, 45, 60, 90, 180])
    parser.add_argument('--severe-translation', '-st', type=float, default=None, choices=[0.09, 0.1, 0.11])
    parser.add_argument('--severe-scaling', '-sc', type=float, default=None, choices=[0.9, 1.1, 1.2])
    parser.add_argument('--num-augmentation', '-na', type=str, default=None, choices=['four', 'one', 'two'])

    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--single-img-infer', action="store_true", default=True)

    # self-supervised augmentation
    parser.add_argument('--angle-list', nargs='+', type=float, default=None)
    parser.add_argument('--translation-list', nargs='+', type=float, default=None)
    parser.add_argument('--scaling-list', nargs='+', type=float, default=None)

    # contraD
    parser.add_argument('--contraD', '-cd', action='store_true', default=False)
    #parser.add_argument('--warmup', action='store_true', default=True)
    #parser.add_argument('--std-flag', action='store_true', default=False)
    #parser.add_argument('--temp', default=0.1, type=float, help='Temperature hyperparameter for contrastive losses')
    parser.add_argument('--weight-simclr-loss', type=float)
    parser.add_argument('--weight-supercon-loss', type=float)

    args = parser.parse_args()
    return args
    
def parse_arguments_centralized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='ixi', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', default='pd', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default='mixed', choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=0.5, choices=[0., 0.1, 0.3, 0.5, 1.])

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--diff-privacy', action='store_true', default=False) 
    parser.add_argument('--identity', action='store_true', default=False)
    parser.add_argument('--reg-gan', action='store_true', default=False)
    parser.add_argument('--fid', action='store_true', default=True)

    # FedMed-ATL 
    parser.add_argument('--atl', action='store_true', default=None, help='indicate whether the atl flag is true or not')
    parser.add_argument('--auxiliary-rotation', '-ar', action='store_true', default=False)
    parser.add_argument('--auxiliary-translation', '-at', action='store_true', default=False)
    parser.add_argument('--auxiliary-scaling', '-as', action='store_true', default=False)

    parser.add_argument('--noise-level', '-nl', type=int, default=None, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--noise-type', '-nt', type=str, default=None, choices=['normal', 'slight', 'severe'])

    # input noise augmentation method
    parser.add_argument('--severe-rotation', '-sr', type=float, default=None, choices=[15, 30, 45, 60, 90, 180])
    parser.add_argument('--severe-translation', '-st', type=float, default=None, choices=[0.09, 0.1, 0.11])
    parser.add_argument('--severe-scaling', '-sc', type=float, default=None, choices=[0.9, 1.1, 1.2])

    # self-supervised augmentation
    parser.add_argument('--angle-list', nargs='+', type=float, default=None)
    parser.add_argument('--translation-list', nargs='+', type=float, default=None)
    parser.add_argument('--scaling-list', nargs='+', type=float, default=None)
    parser.add_argument('--num-augmentation', '-na', type=str, default=None, choices=['four', 'one', 'two'])

    parser.add_argument('--plot-distribution', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--single-img-infer', action='store_true', default=True)


    # contraD
    parser.add_argument('--contraD', '-cd', action='store_true', default=False)
    #parser.add_argument('--warmup', action='store_true', default=False)
    #parser.add_argument('--std-flag', action='store_true', default=False)
    parser.add_argument('--temp', default=None, type=float, help='Temperature hyperparameter for contrastive losses')
    parser.add_argument('--weight-simclr-loss', type=float)
    parser.add_argument('--weight-supercon-loss', type=float)

    args = parser.parse_args()
    return args


def parse_arguments_fid_stats():
    parser = argparse.ArgumentParser("Pre-Calculate Statistics of Images")
    parser.add_argument('--fid-dir', default='./fid_stats', type=str, help='the output path for statistics storage')
    parser.add_argument('--batch-size', type=int, default=50, help='the batchsize for InceptionNetV3')
    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--source-domain', '-s', default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--valid-path', type=str, default=None)

    args = parser.parse_args()   
    return args