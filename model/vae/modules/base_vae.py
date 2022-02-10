from abc import abstractmethod
import torch.nn as nn
import torch

class BaseVAE(nn.Module):
    
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size, current_device):
        raise NotImplementedError

    def generate(self, x: torch.tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


