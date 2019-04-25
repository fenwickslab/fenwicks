import tensorflow as tf
import numpy as np
import functools
from typing import List


def apply_transforms(x, tfms: List):
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)


def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)


def deg2rad(x):
    """
    Converts an angle in degrees to radians.
    :param x: Input angle, in degrees.
    :return: Angle in radians
    """
    return (x * np.pi) / 180
