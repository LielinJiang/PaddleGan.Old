import paddle

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
        return fluid.layers.dropout(x, self.prob, dropout_implementation=self.mode)


class BCEWithLogitsLoss(fluid.dygraph.Layer):
    def __init__(self, weight=None, reduction='mean'):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCEWithLogitsLoss, self).__init__()
        # self.weight = weight
        # self.reduction = reduction
        self.bce_loss = paddle.nn.BCELoss(weight, reduction)

    def forward(self, input, label):
        input = paddle.nn.functional.sigmoid(input, True)
        return self.bce_loss(input, label)


        
