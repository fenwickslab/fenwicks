import tensorflow as tf
import numpy as np
import functools
from typing import List, Callable


def sequential_transforms(x: tf.Tensor, tfms: List[Callable]) -> tf.Tensor:
    """
    Apply a sequence of transform functions to a given input.

    :param x: The input.
    :param tfms: A sequence of transforms, each of which is a function.
    :return: Transformed input.
    """
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)


def parallel_transforms(x: tf.Tensor, trms: List[Callable]) -> List[tf.Tensor]:
    update_func = lambda y: y(x)
    return list(map(update_func, trms))


def random_matmul(mat1: tf.Tensor, mat2: tf.Tensor, p: float) -> tf.Tensor:
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    return tf.cond(choice < p, lambda: tf.matmul(mat1, mat2), lambda: mat1)


def random_transform(x: tf.Tensor, tfm: Callable, p: float) -> tf.Tensor:
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    return tf.cond(choice < p, lambda: tfm(x), lambda: x)


def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)


def deg2rad(x: tf.Tensor) -> tf.Tensor:
    """
    Converts an angle in degrees to radians.
    :param x: Input angle, in degrees.
    :return: Angle in radians
    """
    return (x * np.pi) / 180
