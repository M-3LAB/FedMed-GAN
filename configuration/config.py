import argparse
from asyncio import FastChildWatcher

__all__ = ['parse_argument_bise', 'parse_arguments_federated', 
            'parse_arguments_centralized', 'parse_arguments_fid_stats']


def parse_arguments_bise():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2019', 'brats2021', 'reggandata'])
    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['normal', 'gaussian', 'reg'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'muint', 'unit'])
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--bise-stats', action='store_true', default=None)
    parser.add_argument('--delta-diff', type=float, default=None)
    args = parser.parse_args()
    return args


def parse_arguments_federated():
    parser = argparse.ArgumentParser()
    # federated setting
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default=None)
    parser.add_argument('--num-round', type=int, default=None)

    # centralized setting
    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='flair', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default='mixed', choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=None, choices=[0., 0.1, 0.3, 0.5, 1.])

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=None)
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

    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--single-img-infer', action="store_true", default=False)

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
    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='flair', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)

    parser.add_argument('--data_mode', '-dm', type=str, default=None, choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-paired-weight', '-dpw', type=float, default=None, choices=[0., 0.1, 0.3, 0.5, 1.])

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)

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

    # self-supervised augmentation
    parser.add_argument('--angle-list', nargs='+', type=float, default=None)
    parser.add_argument('--translation-list', nargs='+', type=float, default=None)
    parser.add_argument('--scaling-list', nargs='+', type=float, default=None)
    parser.add_argument('--num-augmentation', '-na', type=str, default=None, choices=['four', 'one', 'two'])

    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--single-img-infer', action='store_true', default=False)


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