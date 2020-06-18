import paddle
import paddle.nn as nn
import functools

from ...modules.nn import ReflectionPad2d, LeakyReLU, Tanh, Dropout, Conv2DTranspose, Conv2D
from ...modules.norm import build_norm_layer
from .builder import GENERATORS


@GENERATORS.register()
class UnetGenerator(paddle.fluid.dygraph.Layer):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type='batch', use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        # tmp = self.model._sub_layers['model'][0](input)
        # tmp1 = self.model._sub_layers['model'][1](tmp)
        # tmp2 = self.model._sub_layers['model'][2](tmp1)
        # import pickle
        # pickle.dump(tmp2.numpy(), open('/workspace/notebook/align_pix2pix/tmp2-pd.pkl', 'wb'))
        # tmp3 = self.model._sub_layers['model'][3](tmp2)
        # pickle.dump(tmp3.numpy(), open('/workspace/notebook/align_pix2pix/tmp3-pd.pkl', 'wb'))
        # tmp4 = self.model._sub_layers['model'][4](tmp3)
        return self.model(input)


class UnetSkipConnectionBlock(paddle.fluid.dygraph.Layer):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm
        if input_nc is None:
            input_nc = outer_nc
        downconv = Conv2D(input_nc, inner_nc, filter_size=4,
                             stride=2, padding=1, bias_attr=use_bias)
        downrelu = LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = Conv2DTranspose(inner_nc * 2, outer_nc,
                                        filter_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = Conv2DTranspose(inner_nc, outer_nc,
                                        filter_size=4, stride=2,
                                        padding=1, bias_attr=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = Conv2DTranspose(inner_nc * 2, outer_nc,
                                        filter_size=4, stride=2,
                                        padding=1, bias_attr=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return paddle.concat([x, self.model(x)], 1)
