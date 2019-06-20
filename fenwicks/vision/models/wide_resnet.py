from ...imports import *
from ... import layers


class WRNBlock(layers.Sequential):
    def __init__(self, c, strides, kernel_initializer: Union[str, Callable] = 'glorot_uniform', bn_mom: float = 0.99,
                 bn_eps: float = 0.001):
        super().__init__()

        main = layers.Parallel(do_add=True)
        backbone = layers.Sequential()
        backbone.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))
        backbone.add(
            layers.ConvBN(c, strides=strides, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))
        backbone.add(
            tf.keras.layers.Conv2D(c, 3, kernel_initializer=kernel_initializer, padding='same', use_bias=False))
        backbone.add(layers.Scaling(0.2))

        main.add(backbone)
        main.add(tf.keras.layers.Conv2D(c, 3, kernel_initializer=kernel_initializer, padding='same', use_bias=False))

        self.add(layers.BNRelu(bn_mom, bn_eps))
        self.add(main)


class WRNGroup(layers.Sequential):
    def __init__(self, n_blocks, c, strides, kernel_initializer: Union[str, Callable] = 'glorot_uniform',
                 bn_mom: float = 0.99, bn_eps: float = 0.001):
        super().__init__()
        for i in range(n_blocks):
            self.add(WRNBlock(c, strides if i == 0 else 1, kernel_initializer, bn_mom, bn_eps))


class WideResNet(layers.Sequential):
    def __init__(self, n_groups: int, n_blocks_per_group: int, n_classes: int, k: int = 1, c: int = 16,
                 kernel_initializer='glorot_uniform', bn_mom=0.99, bn_eps=0.001):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(c, 3, kernel_initializer=kernel_initializer, padding='same', use_bias=False))

        for i in range(n_groups):
            n_channels = c * (2 ** i) * k
            self.add(WRNGroup(n_blocks_per_group, n_channels, (1 if i == 0 else 2), kernel_initializer, bn_mom, bn_eps))

        self.add(layers.BNRelu(bn_mom, bn_eps))
        self.add(tf.keras.layers.GlobalAvgPool2D())
        self.add(tf.keras.layers.Flatten())
        self.add(layers.Classifier(n_classes, kernel_initializer=kernel_initializer))


def wrn_22():
    return WideResNet(n_groups=3, n_blocks_per_group=3, n_classes=10, k=6)
