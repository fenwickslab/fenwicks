import tensorflow as tf
import numpy as np
import random
import functools
from typing import List, Callable, Tuple, Dict


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


def shuffle_lists(list1: List, list2: List) -> Tuple[List, List]:
    c = list(zip(list1, list2))
    random.shuffle(c)
    list1, list2 = zip(*c)
    return list(list1), list(list2)


def get_shape_list(x: tf.Tensor) -> List:
    shape = x.shape.as_list()

    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(x)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(x: tf.Tensor) -> tf.Tensor:
    return x if x.shape.ndims == 2 else tf.reshape(x, [-1, (x.shape[-1])])


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    return tf.reshape(output_tensor, orig_shape_list[0:-1] + [(output_shape[-1])])


def flatten_recursive(item) -> List:
    output = []
    if isinstance(item, list):
        output.extend(item)
    elif isinstance(item, tuple):
        output.extend(list(item))
    elif isinstance(item, dict):
        for _, v in item.items():
            output.append(v)
    else:
        return [item]

    flat_output = []
    for x in output:
        flat_output.extend(flatten_recursive(x))
    return flat_output


def inverse_dict(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def convert_by_dict(d: Dict, items: List) -> List:
    return list(map(lambda x: d[x], items))


def get_node_names() -> List[str]:
    return [n.name for n in tf.get_default_graph().as_graph_def().node]
