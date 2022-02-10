import torch

__all__ = ['GaussianNoise']

class BaseNoise(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass

    def __repr__(self, **kwargs):
        pass
class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        #TODO: Restricted Region pixel != 0
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)