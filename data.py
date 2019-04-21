import tensorflow as tf
from typing import List
from . import core


def tfexample_raw_parser(tfexample: tf.train.Example, has_label: bool = True):
    """
    Parse a given TFExample containing an (image, label) pair, whose contents are undefined.
    :param tfexample: An input TFExample.
    :param has_label: Whether the input TFExample contains a label. Default: True.
    :return: Parsed image (and optionally label) Tensors.
    """

    if has_label:
        feat_dict = {'image': tf.FixedLenFeature([], tf.string),
                     'label': tf.FixedLenFeature([], tf.int64)}
    else:
        feat_dict = {'image': tf.FixedLenFeature([], tf.string)}

    feat = tf.parse_single_example(tfexample, features=feat_dict)

    if has_label:
        return feat['image'], feat['label']
    else:
        return feat['image']


# todo: add dtype as a parameter.
def tfexample_image_parser(tfexample: tf.train.Example, tfms: List = None, has_label: bool = True):
    """
    Parse a given TFExample containing an encoded image (such as JPEG) and a label. Then apply the given sequence of
    transformations.

    :param tfexample: An input TFExample.
    :param tfms: A sequence of transforms.
    :param has_label: Whether the input TFExample contains a label. Default: True.
    :return: Parsed image (and optionally also label) Tensors.
    """

    parsed_example = tfexample_raw_parser(tfexample, has_label)

    if has_label:
        x, y = parsed_example
    else:
        x = parsed_example

    x = tf.image.decode_image(x, channels=3, dtype=tf.float32)
    if tfms is not None:
        x = core.apply_transforms(x, tfms)

    if has_label:
        return x, y
    else:
        return x
