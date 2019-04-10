import tensorflow as tf
from typing import List
import numpy as np
import math


def distort_color(x: tf.Tensor, cb_distortion_range: float = 0.1, cr_distortion_range: float = 0.1) -> tf.Tensor:
    br_delta = tf.random.uniform([], -32. / 255., 32. / 255.)
    cb_factor = tf.random.uniform([], -cb_distortion_range, cb_distortion_range)
    cr_factor = tf.random.uniform([], -cr_distortion_range, cr_distortion_range)

    channels = tf.split(axis=2, num_or_size_splits=3, value=x)
    red_offset = 1.402 * cr_factor + br_delta
    green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
    blue_offset = 1.772 * cb_factor + br_delta
    channels[0] += red_offset
    channels[1] += green_offset
    channels[2] += blue_offset
    x = tf.concat(axis=2, values=channels)
    x = tf.clip_by_value(x, 0., 1.)
    return x


def distorted_bbox_crop(x: tf.Tensor, min_object_covered: float = 0.1, aspect_ratio_range=(3. / 4., 4. / 3.),
                        area_range=(0.05, 1.0), max_attempts: int = 100) -> tf.Tensor:
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_sz, _ = tf.image.sample_distorted_bounding_box(tf.shape(x), bounding_boxes=bbox,
                                                                    min_object_covered=min_object_covered,
                                                                    aspect_ratio_range=aspect_ratio_range,
                                                                    area_range=area_range,
                                                                    max_attempts=max_attempts,
                                                                    use_image_if_no_bounding_boxes=True)
    x = tf.slice(x, bbox_begin, bbox_sz)
    return x


def random_crop(x: tf.Tensor, boxes, scales) -> tf.Tensor:
    # Create different crops for an image
    crops = tf.image.crop_and_resize([x], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
    # Return a random crop
    return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


def random_zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x, boxes, scales))


def random_color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def random_flip(x: tf.Tensor, vertical_flip: bool = False) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    if vertical_flip:
        x = tf.image.random_flip_up_down(x)
    return x


def random_rotate_90(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def random_rotate(x: tf.Tensor, max_deg: float = 45) -> tf.Tensor:
    deg = tf.random_uniform(shape=[], minval=-max_deg, maxval=max_deg, dtype=tf.float32)
    return tf.contrib.image.rotate(x, math.radians(deg))


def random_translate(x: tf.Tensor, max_translation: int = 10) -> tf.Tensor:
    tl = tf.random_uniform(shape=[2], minval=-max_translation, maxval=max_translation, dtype=tf.int32)
    return tf.contrib.image.translate(x, translation=[tl[0], tl[1]])


def ramdom_pad_crop(x: tf.Tensor, pad_size: int) -> tf.Tensor:
    shape = tf.shape(x)
    x = tf.pad(x, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='reflect')
    x = tf.random_crop(x, [shape[0], shape[1], 3])
    return x


def imagenet_normalize_tf(x: tf.Tensor) -> tf.Tensor:
    return (x - 0.5) * 2.0


def imagenet_normalize_pytorch(x: tf.Tensor) -> tf.Tensor:
    return (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


def imagenet_normalize_caffe(x: tf.Tensor) -> tf.Tensor:
    return x[..., ::-1] * 255 - [103.939, 116.779, 123.68]


def get_train_transforms(h: int, w: int, normalizer=imagenet_normalize_tf) -> List:
    return [distorted_bbox_crop,
            lambda x: x.set_shape([None, None, 3]) or x,
            lambda x: tf.image.resize_images(x, [h, w]),
            tf.image.random_flip_left_right,
            distort_color,
            normalizer,
            ]


def get_eval_transforms(h: int, w: int, center_frac: float = 1.0, normalizer=imagenet_normalize_tf) -> List:
    return [lambda x: tf.image.central_crop(x, central_fraction=center_frac),
            lambda x: x.set_shape([None, None, 3]) or x,
            lambda x: tf.image.resize_images(x, [h, w]),
            normalizer,
            ]


# def get_cifar10_train_transforms() -> List:
#     return [cifar10_normalize_pytorch,
#             lambda x: tf.pad(x, [[4, 4], [4, 4], [0, 0]], mode='reflect'),
#             lambda x: tf.random_crop(x, [32, 32, 3]),
#             tf.image.random_flip_left_right,
#             ]
