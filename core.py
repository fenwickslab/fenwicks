import tensorflow as tf
import functools
from typing import List


def apply_transforms(x, tfms: List):
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)


def replace_slice(input_, replacement, begin):
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)
