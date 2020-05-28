from paddle.fluid.dygraph import Layer
from paddle import fluid


class ReflectionPad2d(Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")


class LeakyReLU(Layer):
    def __init__(self, alpha, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return fluid.layers.tanh(x)


class Dropout(Layer):
    def __init__(self, prob, mode='upscale_in_train'):
        super(Dropout, self).__init__()
        self.prob = prob
        self.mode = mode

    def forward(self, x):
        return fluid.layers.dropout(x, self.prob, downgrade_in_infer=self.mode)
