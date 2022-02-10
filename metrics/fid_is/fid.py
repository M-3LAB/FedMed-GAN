from metrics.fid_is.common import get_inception_feature
import numpy as np
import os
from scipy import linalg

__all__ = ['get_stats']

def get_stats(data_loader, batch_size, output_path, source_domain, target_domain, device):

    acts, = get_inception_feature(data_loader, batch_size, dims_list=[2048], 
                                  target_domain=target_domain,
                                  device=device)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    output = os.path.join(output_path, f'{source_domain}_{target_domain}_fid_stats')
    np.savez_compressed(output, mu=mu, sigma=sigma)

def calculate_fid(acts, mu2, sigma2, eps=1e-6):
    """
    Args:
        acts: activation result from InceptionV3
        mu2: The sample mean over activations, precalculated on an reference
             data set.
        sigma2: The covariance matrix over activations, precalculated on an
            reference data set.
        eps: prevent covmean from being singular matrix
    """
    mu1 = np.mean(acts, axis=0)
    sigma1 = np.cov(acts, rowvar=False)
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)


    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = (diff.dot(diff) +
           np.trace(sigma1) +
           np.trace(sigma2) -
           2 * tr_covmean)
    return fid 