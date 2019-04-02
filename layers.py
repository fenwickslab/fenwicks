import tensorflow as tf
from typing import List


class GlobalPools(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        return tf.keras.layers.concatenate([self.gmp(x), self.gap(x)])


class DenseBlk(tf.keras.models.Model):
    def __init__(self, c: int, drop_rate: float):
        super().__init__()
        self.dense = tf.keras.layers.Dense(c, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x):
        return self.dropout(tf.nn.relu(self.bn(self.dense(x))))


class ConvBN(tf.keras.models.Model):
    def __init__(self, c: int, kernel_size: List[int]):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return tf.nn.relu(self.bn(self.conv(x)))
