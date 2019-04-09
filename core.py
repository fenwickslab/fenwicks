import tensorflow as tf
import functools
from typing import List


def apply_transforms(x: tf.Tensor, tfms: List) -> tf.Tensor:
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)
