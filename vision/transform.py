from ..core import *
from .affine import affine_transform


def cutout(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.

    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
    """
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    x = replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
    return x


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


def random_lighting(x: tf.Tensor, max_lighting: float = 0.2) -> tf.Tensor:
    x = tf.image.random_brightness(x, 0.5 * max_lighting)
    x = tf.image.random_contrast(x, 1 - max_lighting, 1 / (1 - max_lighting))
    return x


def random_flip(x: tf.Tensor, flip_vert: bool = False) -> tf.Tensor:
    """
    Randomly flip the input image horizontally, and optionally also vertically.

    :param x: Input image.
    :param flip_vert: Whether to perform vertical flipping. Default: False.
    :return: Transformed image.
    """
    x = tf.image.random_flip_left_right(x)
    if flip_vert:
        x = random_rotate_90(x)
    return x


def random_rotate_90(x: tf.Tensor) -> tf.Tensor:
    """
    Randomly rotate the input image by either 0, 90, 180 or 270 degrees.
    :param x: Input image.
    :return: Transformed image.
    """
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def random_rotate(x: tf.Tensor, max_deg: float = 10) -> tf.Tensor:
    x = tf.expand_dims(x, 0)

    deg = tf.random_uniform(shape=[], minval=-max_deg, maxval=max_deg, dtype=tf.float32)
    theta = tf.convert_to_tensor([
        [tf.cos(deg2rad(deg)), -tf.sin(deg2rad(deg)), 0],
        [tf.sin(deg2rad(deg)), tf.cos(deg2rad(deg)), 0]
    ])

    result = affine_transform(x, theta)
    result = tf.squeeze(result, [0])
    return result


def random_translate(x: tf.Tensor, max_translation: int = 10) -> tf.Tensor:
    tl = tf.random_uniform(shape=[2], minval=-max_translation, maxval=max_translation, dtype=tf.int32)
    return tf.contrib.image.translate(x, translations=[tl[0], tl[1]])


# todo: get partial
def ramdom_pad_crop(x: tf.Tensor, pad_size: int) -> tf.Tensor:
    """
    Randomly pad the image by `pad_size` at each border (top, bottom, left, right). Then, crop the padded image to its
    original size.

    :param x: Input image.
    :param pad_size: Number of pixels to pad at each border. For example, a 32x32 image padded with 4 pixels becomes a
                     40x40 image. Then, the subsequent cropping step crops the image back to 32x32. Padding is done in
                     `reflect` mode.
    :return: Transformed image.
    """
    shape = tf.shape(x)
    x = tf.pad(x, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='reflect')
    x = tf.random_crop(x, [shape[0], shape[1], 3])
    return x


def imagenet_normalize_tf(x: tf.Tensor) -> tf.Tensor:
    """
    Default Tensorflow image normalization for Keras models pre-trained on ImageNet.

    :param x: Input image. Each pixel must be already scaled to [0, 1].
    :return: Normalized image.
    """
    return (x - 0.5) * 2.0


def imagenet_normalize_pytorch(x: tf.Tensor) -> tf.Tensor:
    """
    Default PyTorch image normalization for Keras models pre-trained on ImageNet.
    :param x: Input image. Each pixel must be already scaled to [0, 1].
    :return: Normalized image.
    """
    return (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


def imagenet_normalize_caffe(x: tf.Tensor) -> tf.Tensor:
    """
    Default Caffe image normalization for Keras models pre-trained on ImageNet.
    :param x: Input image. Each pixel must be already scaled to [0, 1].
    :return: Normalized image.
    """
    return x[..., ::-1] * 255 - [103.939, 116.779, 123.68]


# todo: rename to standard scaler
def normalize(x: tf.Tensor, x_mean, x_std) -> tf.Tensor:
    return (x - x_mean) / x_std


def set_shape(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    x.set_shape([h, w, c])
    return x


def get_transform(func, **kw_args):
    return functools.partial(func, kw_args)


def get_train_transforms(h: int, w: int, flip_vert: bool = False, normalizer=imagenet_normalize_tf) -> List:
    return [distorted_bbox_crop,
            functools.partial(set_shape, h=None, w=None),
            functools.partial(tf.image.resize_images, size=[h, w]),
            functools.partial(random_flip, flip_vert=flip_vert),
            distort_color,
            normalizer,
            ]


def get_eval_transforms(h: int, w: int, center_frac: float = 1.0, normalizer=imagenet_normalize_tf) -> List:
    return [functools.partial(tf.image.central_crop, central_fraction=center_frac),
            functools.partial(set_shape, h=None, w=None),
            functools.partial(tf.image.resize_images, size=[h, w]),
            normalizer,
            ]
