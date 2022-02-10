import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = ['ResUnet']

"""
ResUnet Hyparameter
"""
# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }

norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function

def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None

class Conv(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                         |            ^
                                         |__ResBlcok__| (optional)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, n_blocks=1, init_func=init_func) if use_resnet else None
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x

class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, skip=True, refine=False, pool=True,
                 pool_size=2, **kwargs):
        super(DownBlock, self).__init__()
        self.conv_0 = Conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                           activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(out_channels, out_channels, kernel_size, stride, padding, bias=bias,
                               activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                               use_resnet=use_resnet, **kwargs)
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(dim, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                            use_bias=True)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        Conv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=act,
                                             use_norm=False)
                                        )
        else:
            self.refine = lambda x: x
        self.output = Conv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)
    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x

