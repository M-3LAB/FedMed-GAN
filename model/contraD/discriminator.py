import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LinearDiscriminator', 'MLPDiscriminator', 'Projector']

class LinearDiscriminator(nn.Module):
    def __init__(self, n_features=512, n_classes=1):
        super(LinearDiscriminator, self).__init__()
        self.n_features = n_features
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, input):
        input = torch.sum(input, dim=(2, 3))
        d = self.linear(input)
        return d

class MLPDiscriminator(nn.Module):
    def __init__(self, n_features=512, n_classes=1, d_hidden=128):
        super(MLPDiscriminator, self).__init__()

        self.l1 = nn.Linear(n_features, d_hidden)
        self.l2 = nn.Linear(d_hidden, n_classes)

    def forward(self, input):
        input = torch.sum(input, dim=(2, 3))
        output = self.l1(input)
        features = F.leaky_relu(output, 0.1, inplace=True)
        d = self.l2(features)
        return d

class Projector(nn.Module):
    def __init__(self, n_features=512, d_hidden=128):
        super(Projector, self).__init__()
        self.l1 = nn.Linear(n_features, n_features)
        self.l2 = nn.Linear(n_features, d_hidden)
    
    def forward(self, input):
        input = torch.sum(input, dim=(2, 3))
        output = self.l1(input)
        features = F.leaky_relu(output, 0.1, inplace=True)
        d = self.l2(features)
        return d