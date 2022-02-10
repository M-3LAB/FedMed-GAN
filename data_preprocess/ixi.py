import numpy as np
import os
import argparse
import tqdm
import glob

from common import read_img_sitk

def read_multimodal(src_path, dst_path, series):
    for mode in series:
        print('process: ' + mode)
        files = glob.glob("%s/%s/*" % (src_path, mode))

        for f in tqdm.tqdm(files):
            data = read_img_sitk(f)
            if data.shape[0] > 100:
                name = f.split('/')[-1].replace('-{}.nii.gz'.format(mode), '.npy')
                np.save(dst_path + '/' + mode + '/' + name, data)

def dataset_preprocess(src_path, dst_path):
    series = ['PD', 'T2']  # ['DTI', 'MRA', 'PD', 'T1', 'T2']
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        for mode in series:
            os.mkdir(dst_path + '/' + mode)

    read_multimodal(src_path, dst_path, series)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch IXI")
    parser.add_argument("--data_path", default="/disk1/medical/IXI", nargs='+', type=str, help="path to train data")
    parser.add_argument("--generated_path", default="/disk1/medical/ixi", nargs='+', type=str, help="path to target train data")
    args = parser.parse_args()


    dataset_preprocess(src_path=args.data_path, dst_path=args.generated_path)

