import tensorflow as tf
import math
import numpy as np
from .core import *


class GlobalPools(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        return tf.keras.layers.concatenate([self.gmp(x), self.gap(x)])


class DenseBlk(tf.keras.Model):
    def __init__(self, c: int, drop_rate: float):
        super().__init__()
        self.dense = tf.keras.layers.Dense(c, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x):
        return self.dropout(tf.nn.relu(self.bn(self.dense(x))))


class ConvBN(tf.keras.Model):
    def __init__(self, c: int, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99, bn_eps=0.001):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                           padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps)

    def call(self, x):
        return tf.nn.relu(self.bn(self.conv(x)))


class ConvBlk(tf.keras.Sequential):
    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(
            ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))
        self.add(tf.keras.layers.MaxPooling2D() if pool is None else pool)


class ConvResBlk(ConvBlk):
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
        hh = apply_transforms(h, self.res)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_CONV_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}
