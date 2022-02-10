import torch
import torch.nn as nn
from model.cyclegan.cyclegan import *
from model.common import *
from model.contraD.discriminator import *

__all__ = ['ContraD']

class ContraD(nn.Module):

    def __init__(self, mlp_linear=False, auxiliary_rotation=False,
                 auxiliary_translation=False, auxiliary_scaling=False, 
                 num_augmentation='one'):

        super(ContraD, self).__init__()

        """
        Variable Name should refer to "Training GAN with Stronger Augmentations 
        via Contrastive Discriminator"
        """
        self.auxiliary_rotation = auxiliary_rotation 
        self.auxiliary_translation = auxiliary_translation 
        self.auxiliary_scaling = auxiliary_scaling
        self.num_augmentation = num_augmentation

        if self.num_augmentation == 'four':
            self.num_rot_label = 4 
            self.num_translate_label = 5 
            self.num_scaling_label = 4 

        elif self.num_augmentation == 'one':
            self.num_rot_label = 2 
            self.num_translate_label = 2 
            self.num_scaling_label = 2 
        
        elif self.num_augmentation == 'two':
            self.num_rot_label = 3 
            self.num_translate_label = 3 
            self.num_scaling_label = 3 

        self.mlp_linear = mlp_linear

        # A bunch of convolutions one after another
        model = [nn.Conv2d(1, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
              nn.InstanceNorm2d(512),
              nn.LeakyReLU(0.2, inplace=True)]


        self.model = nn.Sequential(*model) 

        # Final Classification Layer, It also refer as the discriminator head
        if mlp_linear:
            self.discriminator_head = MLPDiscriminator(n_classes=1)
            self.rot_discriminator_head = MLPDiscriminator(n_classes=self.num_rot_label)
            self.translate_discriminator_head = MLPDiscriminator(n_classes=self.num_translate_label)
            self.scale_discriminator_head = MLPDiscriminator(n_classes=self.num_scaling_label)
        else:
            self.discriminator_head = LinearDiscriminator(n_classes=1)
            self.rot_discriminator_head = LinearDiscriminator(n_classes=self.num_rot_label)
            self.translate_discriminator_head = LinearDiscriminator(n_classes=self.num_translate_label)
            self.scale_discriminator_head = LinearDiscriminator(n_classes=self.num_scaling_label)

        #TODO: Fix the bugs for d_penul and d_hidden 
        #TODO: add std_flag situation

        #self.projection_real = nn.Sequential(nn.Linear(512, 512),
        #                                     nn.LeakyReLU(0.1, inplace=True),
        #                                     nn.Linear(512, 128)) 
        #self.projection_fake = nn.Sequential(nn.Linear(512, 512),
        #                                     nn.LeakyReLU(0.1, inplace=True),
        #                                     nn.Linear(512, 128)) 

        self.projection_real = Projector() 
        self.projection_fake = Projector()

    def forward(self, fake_x=None, real_x=None, real_x1=None, real_x2=None, 
                projection_head='real_head', rot_x=None, translate_x=None, 
                scale_x=None, stop_gradient=False):

        #TODO: Stop Gradient Situtation
        """
        real_x1 and real_x2 are two different augmentation method generated from real-x
        For example, real_x1 is rot_real_x and real_x2 is translate_real_x
        """

        if projection_head == 'discriminator_head':

            if fake_x is not None: 
                fake_x = self.model(fake_x)
                gan_logit = self.discriminator_head(fake_x)
                return gan_logit

            elif real_x is not None:
                real_x = self.model(real_x)
                gan_logit = self.discriminator_head(real_x)
                return gan_logit

            elif self.auxiliary_rotation and rot_x is not None:
                rot_x = self.model(rot_x)
                rot_logit = self.rot_discriminator_head(rot_x)
                return rot_logit

            elif self.auxiliary_translation and translate_x is not None:
                translate_x = self.model(translate_x)
                translate_logit = self.translate_discriminator_head(translate_x)
                return translate_logit

            elif self.auxiliary_scaling and scale_x is not None:
                scale_x = self.model(scale_x)
                scale_logit = self.scale_discriminator_head(scale_x)
                return scale_logit
            else:
                raise ValueError('Discriminator Head Only Accept Fake_X and Real_X') 
        
        elif projection_head == 'real_head':

            assert real_x1 is not None
            assert real_x2 is not None

            real_x1 = self.model(real_x1)
            real_x1_logit = self.projection_real(real_x1)

            real_x2 = self.model(real_x2)
            real_x2_logit = self.projection_real(real_x2)

            return real_x1_logit, real_x2_logit
        
        elif projection_head == 'fake_head':

            assert real_x1 is not None
            assert real_x2 is not None
            assert fake_x is not None

            real_x1 = self.model(real_x1)
            real_x1_logit = self.projection_fake(real_x1)

            real_x2 = self.model(real_x2)
            real_x2_logit = self.projection_fake(real_x2)

            fake_x = self.model(fake_x)
            fake_x_logit = self.projection_fake(fake_x)

            return real_x1_logit, real_x2_logit, fake_x_logit
        
        else:
            raise NotImplementedError('Projection Head Not Implemented Yet')
            

    #def penultimate(self, input):
    #    input = input * 2. - 1.
    #    out = self.model(input)
    #    out = minibatch_stddev_layer(out)
    #    out = out.view(out.size(0), -1)

    #    #TODO: Figure out the output dimention of STD Layer
    #    return out