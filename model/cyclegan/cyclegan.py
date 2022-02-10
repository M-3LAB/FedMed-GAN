import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['CycleGen', 'CycleDis', 'UNetDown', 'UNetUp']

class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                               stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class UnetFinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.
    """
    def __init__(self, in_size, out_size):
        super(UnetFinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

class CycleGen(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(CycleGen, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = UnetFinalLayer(128, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)

def discriminator_block(in_filters, out_filters):
    """Return downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class CycleDis(nn.Module):

    def __init__(self, auxiliary_rotation=False, 
                 auxiliary_translation=False, auxiliary_scaling=False,
                 num_augmentation='one'):

        super(CycleDis, self).__init__()

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

        # FCN classification layer
        self.model = nn.Sequential(*model)
        self.fcn = nn.Conv2d(512, 1, 4, padding=1)

        self.fcn_rot = nn.Linear(512, self.num_rot_label)
        self.fcn_translate = nn.Linear(512, self.num_translate_label)
        self.fcn_scaling = nn.Linear(512, self.num_scaling_label)
        #self.softmax = nn.Softmax()

    def forward(self, x=None, rot_x=None, translate_x=None, scale_x=None):
        if self.auxiliary_rotation and rot_x is not None:
            rot_x = self.model(rot_x)
            rot_x = torch.sum(rot_x, dim=(2, 3))
            rot_logits = self.fcn_rot(rot_x) 
            return rot_logits

        elif self.auxiliary_translation and translate_x is not None:
            translate_x = self.model(translate_x)
            translate_x = torch.sum(translate_x, dim=(2, 3))
            translate_logits = self.fcn_translate(translate_x) 
            return translate_logits

        elif self.auxiliary_scaling and scale_x is not None:
            scale_x = self.model(scale_x)
            scaling_x = torch.sum(scale_x, dim=(2, 3))
            scaling_logits = self.fcn_scaling(scaling_x) 
            return scaling_logits
        else: 
            x = self.model(x)
            x = self.fcn(x)
            # Average pooling and flatten
            gan_logit = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
            return gan_logit