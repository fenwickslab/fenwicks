import tensorflow as tf
import math
import numpy as np
from .core import apply_transforms


class GlobalPools(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        return tf.keras.layers.concatenate([self.gmp(x), self.gap(x)])


class DenseBlk(tf.keras.Sequential):
    def __init__(self, c: int, drop_rate: float):
        super().__init__()
        self.add(tf.keras.layers.Dense(c, use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(drop_rate))


class ConvBN(tf.keras.Sequential):
    def __init__(self, c: int, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99, bn_eps=0.001):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                        padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))


class ConvBlk(tf.keras.Sequential):
    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(
            ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))
        if pool is None:
            pool = tf.keras.layers.MaxPooling2D()
        self.add(pool)


class ConvResBlk(ConvBlk):
    def __init__(self, c, pool=None, convs=1, res_convs=2, kernel_size=3, kernel_initializer='glorot_uniform',
                 bn_mom=0.99, bn_eps=0.001):
        super().__init__(c, pool=pool, convs=convs, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         bn_mom=bn_mom, bn_eps=bn_eps)
        self.res = tf.keras.Sequential()
        for i in range(res_convs):
            self.res.add(
                ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))

    def call(self, x):
        h = super().call(x)
        hh = self.res.call(x)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_CONV_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}
