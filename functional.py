from .imports import *


def gelu(x: tf.Tensor) -> tf.Tensor:
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def dropout(x, dropout_prob):
    return x if dropout_prob is None or dropout_prob == 0.0 else tf.nn.dropout(x, 1.0 - dropout_prob)
