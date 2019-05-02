import math
import tensorflow as tf
import numpy as np
from . import core
from typing import Union, Callable


# todo: SequentialLayer
# todo: Parallel
class Sequential(tf.keras.Model):
    """
    A sequential model (or composite layer), which executes its internal layers sequentially in the same order they are
    added. Sequential can be initialized as an empty model / layer. More layers can be added later on.
    """

    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer):
        self.fw_layers.append(layer)

    def call(self, x):
        return core.apply_transforms(x, self.fw_layers)


class Scaling(tf.keras.layers.Layer):
    """
    Scaling layer, commonly used right before a Softmax activation, since Softmax is sensitive to scaling. It simply
    multiplies its input by a constant weight (not trainable), which is a hyper-parameter.
    """

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def call(self, x):
        return x * self.weight


class GlobalPools(tf.keras.layers.Layer):
    """
    A concatenation of GlobalMaxPooling2D and GlobalAveragiePooling2D.
    """

    def __init__(self):
        super().__init__()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        return tf.keras.layers.concatenate([self.gmp(x), self.gap(x)])


class DenseBN(Sequential):
    """
    A Dense layer followed by BatchNormalization, ReLU activation, and optionally Dropout.
    """

    def __init__(self, c: int, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001, drop_rate: float = 0.0, bn_before_relu=True):
        """
        :param c: number of neurons in the Dense layer.
        :param kernel_initializer: initialization method for the Dense layer.
        :param drop_rate: Dropout rate, i.e., 1-keep_probability. Default: no dropout.
        """
        super().__init__()
        self.add(tf.keras.layers.Dense(c, kernel_initializer=kernel_initializer, use_bias=False))
        bn = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps)
        relu = tf.keras.layers.Activation('relu')

        if bn_before_relu:
            self.add(bn)
            self.add(relu)
        else:
            self.add(relu)
            self.add(bn)

        if drop_rate > 0.0:
            self.add(tf.keras.layers.Dropout(drop_rate))


class Classifier(Sequential):
    def __init__(self, n_classes: int, kernel_initializer: Union[str, Callable] = 'glorot_uniform',
                 weight: float = 1.0):
        super().__init__()
        self.add(tf.keras.layers.Dense(n_classes, kernel_initializer=kernel_initializer, use_bias=False))
        self.add(Scaling(weight))


class ConvBN(Sequential):
    """
    A Conv2D followed by BatchNormalization and ReLU activation.
    """

    def __init__(self, c: int, kernel_size=3, strides=(1, 1), kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=strides,
                                        kernel_initializer=kernel_initializer, padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))
        self.add(tf.keras.layers.Activation('relu'))


class ConvBlk(Sequential):
    """
    A block of `ConvBN` layers, followed by a pooling layer.
    """

    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        for i in range(convs):
            self.add(
                ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))
        self.add(pool or tf.keras.layers.MaxPooling2D())


class ConvResBlk(ConvBlk):
    """
    A `ConvBlk` with additional residual `ConvBN` layers.
    """

    def __init__(self, c, pool=None, convs=1, res_convs=2, kernel_size=3, kernel_initializer='glorot_uniform',
                 bn_mom=0.99, bn_eps=0.001):
        super().__init__(c, pool=pool, convs=convs, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         bn_mom=bn_mom, bn_eps=bn_eps)
        self.res = []
        for i in range(res_convs):
            conv_bn = ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                             bn_eps=bn_eps)
            self.res.append(conv_bn)

    def call(self, inputs):
        h = super().call(inputs)
        hh = core.apply_transforms(h, self.res)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    """
    Initialize a given layer, such as Conv2D or Dense, in the same way as PyTorch.

    Args:
    :param shape: Shape of the weights in the layer to be initialized.
    :param dtype: Data type of the initial weights.
    :param partition_info: Required by Keras. Not used.
    :return: Random weights for a the given layer.
    """
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}


class FastAiHead(Sequential):
    def __init__(self, n_classes: int):
        super().__init__()
        self.add(GlobalPools)
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.BatchNormalization(momentum=PYTORCH_PARAMS['bn_mom'],
                                                    epsilon=PYTORCH_PARAMS['bn_eps']))
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(DenseBN(512, bn_before_relu=False, **PYTORCH_PARAMS))
        self.add(tf.keras.layers.Dropout(0.5))
        self.add(Classifier(n_classes, kernel_initializer=PYTORCH_PARAMS['kernel_initializer']))
