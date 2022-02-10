import torch
import numpy as np
from metrics.fid_is.inception_score import calculate_is
from metrics.fid_is.fid import calculate_fid
from metrics.fid_is.common import get_inception_feature

__all__ = ['mae', 'psnr', 'ssim', 'fid', 'inception_score']

def mae(image_true, image_generated):
    """Compute mean absolute error.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        mse: (float) mean squared error
    """
    return torch.abs(image_true - image_generated).mean()


def psnr(image_true, image_generated):
    """"Compute peak signal-to-noise ratio.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image

    Returns:
        psnr: (float) peak signal-to-noise ratio"""
    mse = ((image_true - image_generated) ** 2).mean().cpu()
    return -10 * np.log10(mse)


def ssim(image_true, image_generated, C1=0.01, C2=0.03):
    """Compute structural similarity index.

    Args:
        image_true: (Tensor) true image
        image_generated: (Tensor) generated image
        C1: (float) variable to stabilize the denominator
        C2: (float) variable to stabilize the denominator

    Returns:
        ssim: (float) mean squared error"""
    mean_true = image_true.mean()
    mean_generated = image_generated.mean()
    std_true = image_true.std()
    std_generated = image_generated.std()
    covariance = (
        (image_true - mean_true) * (image_generated - mean_generated)).mean()

    numerator = (2 * mean_true * mean_generated + C1) * (2 * covariance + C2)
    denominator = ((mean_true ** 2 + mean_generated ** 2 + C1) *
                   (std_true ** 2 + std_generated ** 2 + C2))
    return numerator / denominator

def fid(image_generated, batch_size, target_domain, fid_stats, device):

    acts, = get_inception_feature(image_generated, batch_size, dims_list=[2048],
                                  target_domain=target_domain, device=device)

    f = np.load(fid_stats)
    mu2, sigma2 = f['mu'][:], f['sigma'][:]
    f.close() 
    fid = calculate_fid(acts, mu2, sigma2)
    return fid


def inception_score(image_generated, batch_size, target_domain, device):
    """
    Args:
        batch_size: the batch_size for inceptionV3 
    """
    probs, = get_inception_feature(image_generated, batch_size, dims_list=[1008], 
                                   target_domain=target_domain, device=device)

    inception_score, std = calculate_is(probs, splits=10)
    return (inception_score, std)
