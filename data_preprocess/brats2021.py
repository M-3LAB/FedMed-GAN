import argparse
import os

from common import *
from brats2019 import *


if __name__ == '__main__':

    """
    # source dataset root: BraTS2021
    # generated dataset root: brats2021
    """
    parser = argparse.ArgumentParser(description="PyTorch BraTS2021")

    parser.add_argument("--train_path", default="/disk1/medical/BraTS2021/training", nargs='+', type=str, help="path to train data")
    parser.add_argument("--test_path", default="/disk1/medical/BraTS2021/validation", nargs='+', type=str, help="path to test data") 
    parser.add_argument("--root_generated_path", default="/disk1/medical/brats2021", nargs='+', type=str, help="path to target root")
    parser.add_argument("--train_generated_path", default="/disk1/medical/brats2021/training", nargs='+', type=str, help="path to target train data")
    parser.add_argument("--test_generated_path", default="/disk1/medical/brats2021/validation", nargs='+', type=str, help="path to target test data") 
    
    args = parser.parse_args()

    for dir in [args.root_generated_path, args.train_generated_path, args.test_generated_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    dataset_preprocess(src_path=args.train_path, dst_path=args.train_generated_path)
    dataset_preprocess(src_path=args.test_path, dst_path=args.test_generated_path)



