import paddle
import paddle.nn as nn
import functools

from ...modules.nn import ReflectionPad2d, LeakyReLU, Tanh, Dropout, BCEWithLogitsLoss, Conv2DTranspose, Conv2D, Pad2D, MSELoss
from ...modules.norm import build_norm_layer

from .builder import GENERATORS


@GENERATORS.register()
class ResnetGenerator(paddle.fluid.dygraph.Layer):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_type='instance', use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm

        print('norm layer:', norm_layer, 'use bias:', use_bias)

        model = [ReflectionPad2d(3),
                 nn.Conv2D(input_nc, ngf, filter_size=7, padding=0, bias_attr=use_bias),
                #  nn.nn.Conv2D(input_nc, ngf, filter_size=7, padding=0, bias_attr=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                      nn.Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=1, bias_attr=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                      nn.Conv2DTranspose(ngf * mult, int(ngf * mult / 2),
                                         filter_size=3, stride=2,
                                         padding=1, #output_padding=1,
                                        #  padding='same', #output_padding=1,
                                         bias_attr=use_bias),
                      Pad2D(paddings=[0, 1, 0, 1], mode='constant', pad_value=0.0),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU()]
        model += [ReflectionPad2d(3)]
        model += [nn.Conv2D(ngf, output_nc, filter_size=7, padding=0)]
        model += [Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        # x = paddle.fluid.layers.clip(x, min=-1.0, max=1.0)
        return self.model(x)


class ResnetBlock(paddle.fluid.dygraph.Layer):
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
            conv_block += [ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2D(dim, dim, filter_size=3, padding=p, bias_attr=use_bias), norm_layer(dim), nn.ReLU()]
        if use_dropout:
            conv_block += [Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2D(dim, dim, filter_size=3, padding=p, bias_attr=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
