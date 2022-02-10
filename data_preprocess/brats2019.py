import argparse
import numpy as np
import os
import tqdm
from common import * 



def read_multimodal(data_path, series, annotation_path=None, read_annotation=True):
    suffixes = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']

    affine = read_nii_header(os.path.join(data_path, series, series + suffixes[0])).affine
    files = [read_img_sitk(os.path.join(data_path, series, series + s)) for s in suffixes]
    data = np.stack(files, axis=0).astype(np.float32)
    annotation = None
    if read_annotation:
        p = os.path.join(data_path, series, series + '_seg.nii.gz')
        if annotation_path is not None and not os.path.isfile(p):
            p = os.path.join(annotation_path, series + '.nii.gz')
        annotation = read_nii(p)
        annotation[annotation == 4] = 3

    return data, annotation, affine

def save_image_to_numpy(data, path, name):
    np.save(path + '/T1/' + name, data[0, :, :, :])
    np.save(path + '/T1CE/' + name, data[1, :, :,:])
    np.save(path + '/T2/' + name, data[2, :, :,:])
    np.save(path + '/FLAIR/' + name, data[3, :, :,:])

def dataset_preprocess(src_path, dst_path):
    series = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]
    for mode in ['T1', 'T1CE', 'T2', 'FLAIR']:
        if not os.path.exists(dst_path + '/' + mode):
            os.mkdir(dst_path + '/' + mode)

    path_series = [(src_path, s) for s in series]

    for p, f in tqdm.tqdm(path_series):
        data, annotation, affine = read_multimodal(p, f, annotation_path=None, read_annotation=None)
        save_image_to_numpy(data, dst_path, f) # choose 100th slice


if __name__ == '__main__':
    """
    # source dataset root: BraTS2019
    # generated dataset root: brats2019
    """
    parser = argparse.ArgumentParser(description="PyTorch BraTS2019")
    parser.add_argument("--mode", default="LGG", nargs='+', type=str, help="choose categories")
    parser.add_argument("--train_path", default="/disk1/medical/BraTS2019/training", nargs='+', type=str, help="path to train data")
    parser.add_argument("--test_path", default="/disk1/medical/BraTS2019/validation", nargs='+', type=str, help="path to test data") 
    parser.add_argument("--root_generated_path", default="/disk1/medical/brats2019", nargs='+', type=str, help="path to target root")
    parser.add_argument("--train_generated_path", default="/disk1/medical/brats2019/training", nargs='+', type=str, help="path to target train data")
    parser.add_argument("--test_generated_path", default="/disk1/medical/brats2019/validation", nargs='+', type=str, help="path to target test data") 
    args = parser.parse_args()

    
    for dir in [args.root_generated_path, args.train_generated_path, args.test_generated_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # training dataset
    args.mode = 'LGG'
    if not os.path.exists(args.train_generated_path + '/' + args.mode):
        os.makedirs(args.train_generated_path + '/' + args.mode)
    dataset_preprocess(src_path=args.train_path + '/' + args.mode, dst_path=args.train_generated_path + '/' + args.mode)

    args.mode = 'HGG'
    if not os.path.exists(args.train_generated_path + '/' + args.mode):
        os.makedirs(args.train_generated_path + '/' + args.mode)
    dataset_preprocess(src_path=args.train_path + '/' + args.mode, dst_path=args.train_generated_path + '/' + args.mode)

    args.mode = 'ALL'
    if not os.path.exists(args.train_generated_path + '/' + args.mode):
        os.makedirs(args.train_generated_path + '/' + args.mode)
    os.system('cp -r {} {}'.format(args.train_generated_path + '/LGG/*', args.train_generated_path + '/' + args.mode))
    os.system('cp -r {} {}'.format(args.train_generated_path + '/HGG/*', args.train_generated_path + '/' + args.mode))
    
    # test dataset
    dataset_preprocess(src_path=args.test_path, dst_path=args.test_generated_path)