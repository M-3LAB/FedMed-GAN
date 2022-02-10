import numpy as np
import torch
import sys
sys.path.append('./')
from data_preprocess.common import *
from model.FT.fourier_transform import * 
import matplotlib.pyplot as plt
import kornia.geometry.transform as kt 

__all__ = ['normalise', 'torch_normalise', 'torch_2d_normalise']

def deformation_map():
    pass

def scaling_kspace(k_space):
    k_space_abs = np.abs(k_space)
    scaling = np.power(10., -3)
    np.log1p(k_space_abs * scaling, out=k_space_abs)
    normalise(k_space_abs)
    k_space_abs = np.require(k_space_abs, np.uint8) 
    #print(f'k_space_abs max pixel: {np.max(k_space_abs)}')
    return k_space_abs

def torch_scaling_kspace(k_space):
    k_space_abs = torch.abs(k_space)
    scaling = 0.01
    torch.log1p(scaling * k_space_abs, out=k_space_abs)
    torch_2d_normalise(k_space_abs)
    k_space_abs = k_space_abs.to(torch.uint8)
    return k_space_abs

def ixi_reader(file_path, index=80):
    mri_slices = read_img_sitk(file_path)
    mri = mri_slices[index]
    return mri

def brats_reader(file_path, index=80):
    pass

def to_tensor(mri_img):
    mri_img = np.array(mri_img)
    mri_img = torch.from_numpy(mri_img)
    return mri_img

def to_bchw_tensor(mri_img):
    mri_img = np.array(mri_img)
    mri_img = torch.from_numpy(mri_img)
    mri_img = torch.unsqueeze(torch.unsqueeze(mri_img, 0), 0) 
    return mri_img 

def bchw_tensor_to_img(mri_tensor):
    mri_tensor = torch.squeeze(mri_tensor)
    return mri_tensor

def torch_normalise(f):
    """ 
    Normalises torch tensor by "streching" all values to be between 0-255.
    Parameters:
        f (torch tensor): BCHW, C = 1 due to the characteristics of medicial image    
    """
    for i in range(f.size()[0]):
        fmax = float(torch.max(f[i, 0, :]))
        fmin = float(torch.min(f[i, 0, :]))
        if fmax != fmin:
            coeff = fmax - fmin
            f[i,0, :] = torch.floor((f[i,0, :] - fmin) / coeff * 255.)

def torch_2d_normalise(f): 
    """ 
    Normalises 2D torch tensor by "streching" all values to be between 0-255.
    Parameters:
        f (2D torch tensor): 2D torch tensor
    """
    fmax = float(torch.max(f))
    fmin = float(torch.min(f))
    if fmax != fmin:
        coeff = fmax - fmin
        f[:] = torch.floor((f[:] - fmin) / coeff * 255.)

def normalise(f: np.ndarray):
    """ 
    Normalises array by "streching" all values to be between 0-255.
    Parameters:
        f (np.ndarray): input array
    """
    fmin = float(f.min())
    fmax = float(f.max())
    if fmax != fmin:
        coeff = fmax - fmin
        f[:] = np.floor((f[:] - fmin) / coeff * 255.)

if __name__ == '__main__': 

    t2 = '/disk/medical/BraTS2021/validation/BraTS2021_01679/BraTS2021_01679_t2.nii.gz'
    t1 = '/disk/medical/BraTS2021/validation/BraTS2021_01679/BraTS2021_01679_t1.nii.gz'
    flair = '/disk/medical/BraTS2021/validation/BraTS2021_01679/BraTS2021_01679_flair.nii.gz'
    t1_mri = ixi_reader(t1)
    t2_mri = ixi_reader(t2)
    flair_mri = ixi_reader(flair)
    t2_mri = ixi_reader(t2)

    t2_mri = to_bchw_tensor(t2_mri) 
    angle = torch.tensor([-5.])

    translation = torch.tensor([[-15., -15.]])
    scale = torch.tensor([[1.1, 1.1]])

    rot_t2_mri = kt.rotate(t2_mri, angle)
    translate_t2_mri = kt.translate(t2_mri, translation)
    scale_t2_mri = kt.scale(t2_mri, scale)

    torch_normalise(rot_t2_mri)
    torch_normalise(translate_t2_mri)
    torch_normalise(scale_t2_mri)
    torch_normalise(t2_mri)
    rot_t2_mri = torch.squeeze(rot_t2_mri)
    t2_mri = torch.squeeze(t2_mri)
    translate_t2_mri = torch.squeeze(translate_t2_mri)
    scale_t2_mri = torch.squeeze(scale_t2_mri)


    import cv2
    cv2.imwrite('./legacy_code/vis/t1_mri.png', np.array(t1_mri)/2)
    cv2.imwrite('./legacy_code/vis/flair_mri.png', np.array(flair_mri)/2)
    cv2.imwrite('./legacy_code/vis/t2_mri.png', np.array(t2_mri))
    cv2.imwrite('./legacy_code/vis/rot_t2_mri.png', np.array(rot_t2_mri))
    cv2.imwrite('./legacy_code/vis/translate_t2_mri.png', np.array(translate_t2_mri))
    cv2.imwrite('./legacy_code/vis/scale_t2_mri.png', np.array(scale_t2_mri))

    print(f'success')
