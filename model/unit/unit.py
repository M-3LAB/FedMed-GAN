import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ['Encoder', 'Generator', 'Discriminator', 'ResidualBlock', 'compute_kl']

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu, device):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape))).to(device)
        return z + mu

    def forward(self, x, device):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu, device)
        return mu, z


class Generator(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape, auxiliary_rotation=False, auxiliary_translation=False,
                auxiliary_scaling=False, num_rot_label=4, num_translate_label=5, num_scaling_label=4):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.auxiliary_rotation = auxiliary_rotation 
        self.auxiliary_translation = auxiliary_translation 
        self.auxiliary_scaling = auxiliary_scaling
        self.num_rot_label = num_rot_label
        self.num_translate_label = num_translate_label
        self.num_scaling_label = num_scaling_label

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.cnv = nn.Sequential(
            nn.Conv2d(512, 1, 3, padding=1)
        )

        self.fcn_rot = nn.Linear(512, self.num_rot_label)
        self.fcn_translate = nn.Linear(512, self.num_translate_label)
        self.fcn_scaling = nn.Linear(512, self.num_scaling_label)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x=None, rot_x=None, translate_x=None, scale_x=None):
        if self.auxiliary_rotation and rot_x is not None:
            rot = torch.sum(self.models(rot_x), dim=(2, 3))
            rot_logits = self.fcn_rot(rot)
            return rot_logits
        elif self.auxiliary_translation and translate_x is not None:
            translate = torch.sum(self.models(translate_x), dim=(2, 3))
            translate_logits = self.fcn_translate(translate) 
            return translate_logits
        elif self.auxiliary_scaling and scale_x is not None:
            scale = torch.sum(self.models(scale_x), dim=(2, 3))
            scaling_logits = self.fcn_scaling(scale) 
            return scaling_logits
        else: 
            return self.cnv(self.models(x))