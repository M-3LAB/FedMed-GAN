import torch
import torch.nn as nn
from metrics.fid_is.inception import InceptionV3
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

__all__ = ['get_inception_feature']

def get_inception_feature(images, batch_size, dims_list, target_domain, device):
    """
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
            must be float tensor of range [0, 1].
        batch_size: the batch size for InceptionNetV3
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
            available dimension.
        device: the torch device which is used to calculate inception feature
    """
    assert all(dim in InceptionV3.BLOCK_INDEX_BY_DIM for dim in dims_list) 

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [InceptionV3.BLOCK_INDEX_BY_DIM[dim] for dim in dims_list]
    model = InceptionV3(block_idxs).to(device)
    model.eval()
    features = [np.empty((num_images, dim)) for dim in dims_list]

    pbar = tqdm(total=num_images, dynamic_ncols=True, leave=False,
                disable=False, desc="get_inception_feature")
    
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
            batch_len = len(batch_images[target_domain])
        else:
            batch_images = images[start: start + batch_size]
            batch_len = len(batch_images)

        end = start + batch_len 

        # calculate inception feature
        if is_dataloader:
            batch_images = batch_images[target_domain]

        #Solve the mis-dimensional problem of InceptionV3
        batch_images = torch.cat((batch_images, batch_images, batch_images), 1)
        batch_images = batch_images.to(device)

        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims_list):
                feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(batch_len)

    pbar.close()
    return features
