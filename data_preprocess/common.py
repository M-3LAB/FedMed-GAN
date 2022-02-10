import numpy as np
import SimpleITK as sitk
import nibabel as nib

__all__ = ['read_img_sitk', 'read_nii', 'read_nii_header']

def read_img_sitk(img):
    inputImage = sitk.ReadImage(img)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    image = sitk.GetArrayFromImage(inputImage)
    return image

def read_nii(filename):
    image = nib.load(filename)
    return np.array(image.get_data())

def read_nii_header(filename):
    return nib.load(filename)