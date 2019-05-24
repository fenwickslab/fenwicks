from ..imports import *

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
    x = core.replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
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


def random_flip(x: tf.Tensor, flip_vert: bool = False) -> tf.Tensor:
    """
    Randomly flip the input image horizontally, and optionally also vertically, which is implemented as 90-degree
    rotations.

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


def apply_affine_mat(x: tf.Tensor, mat: tf.Tensor, do_reflect: bool = False) -> tf.Tensor:
    mat = tf.reshape(mat, [-1])[:6]
    x = tf.expand_dims(x, 0)
    x = affine_transform(x, mat, do_reflect)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = tf.squeeze(x, [0])
    return x


def apply_affine_mats(x: tf.Tensor, mats: List[tf.Tensor], ps: List[float]) -> tf.Tensor:
    m = tf.convert_to_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    for mat, p in zip(mats, ps):
        m = core.random_matmul(m, mat, p)

    return apply_affine_mat(x, m)


def random_rotate_matrix(max_deg: float = 10) -> tf.Tensor:
    deg = tf.random_uniform(shape=[], minval=-max_deg, maxval=max_deg, dtype=tf.float32)
    rad = core.deg2rad(deg)
    return tf.convert_to_tensor([[tf.cos(rad), -tf.sin(rad), 0], [tf.sin(rad), tf.cos(rad), 0], [0, 0, 1]])


def random_rotate(x: tf.Tensor, max_rot_deg: float = 10) -> tf.Tensor:
    mat = random_rotate_matrix(max_rot_deg)
    return apply_affine_mat(x, mat)


def random_zoom_matrix(max_zoom: float = 1.1, row_pct: float = 0.5, col_pct: float = 0.5) -> tf.Tensor:
    scale = tf.random_uniform(shape=[], minval=1.0, maxval=max_zoom, dtype=tf.float32)
    s = 1 - 1 / scale
    col_c = s * (2 * col_pct - 1)
    row_c = s * (2 * row_pct - 1)
    return tf.convert_to_tensor([[1 / scale, 0, col_c], [0, 1 / scale, row_c], [0, 0, 1]])


def random_zoom(x: tf.Tensor, max_zoom: float = 1.1, row_pct: float = 0.5, col_pct: float = 0.5) -> tf.Tensor:
    mat = random_zoom_matrix(max_zoom, row_pct, col_pct)
    return apply_affine_mat(x, mat)


def random_shear_matrix(max_shear_deg: float = 10) -> tf.Tensor:
    deg = tf.random_uniform(shape=[], minval=-max_shear_deg, maxval=max_shear_deg, dtype=tf.float32)
    rad = core.deg2rad(deg)
    return tf.convert_to_tensor([[1, -tf.sin(rad), 0], [0, tf.cos(rad), 0], [0, 0, 1]])


def random_shear(x: tf.Tensor, max_shear_deg: float = 10) -> tf.Tensor:
    mat = random_shear_matrix(max_shear_deg)
    return apply_affine_mat(x, mat)


def random_shift_matrix(wrg: float = 0.1, hrg: float = 0.1) -> tf.Tensor:
    tx = tf.random_uniform(shape=[], minval=-hrg, maxval=hrg, dtype=tf.float32)
    ty = tf.random_uniform(shape=[], minval=-wrg, maxval=wrg, dtype=tf.float32)
    return tf.convert_to_tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def random_shift(x: tf.Tensor, wrg: float = 0.1, hrg: float = 0.1) -> tf.Tensor:
    mat = random_shift_matrix(wrg, hrg)
    return apply_affine_mat(x, mat)


def random_dihedral_matrix() -> tf.Tensor:
    k = tf.random_uniform(shape=[], minval=0, maxval=8, dtype=tf.int32)
    x = tf.bitwise.bitwise_and(k, 1) * -2 + 1
    y = tf.bitwise.bitwise_and(k, 2) * -2 + 1
    return tf.cond(tf.bitwise.bitwise_and(k, 4) > 0, lambda: tf.convert_to_tensor([[0, x, 0.], [y, 0, 0], [0, 0, 1.]]),
                   lambda: tf.convert_to_tensor([[x, 0, 0.], [0, y, 0], [0, 0, 1.]]))


def flip_matrix() -> tf.Tensor:
    return tf.convert_to_tensor([[-1, 0, 0.], [0, 1, 0], [0, 0, 1.]])


def random_dihedral(x: tf.Tensor) -> tf.Tensor:
    mat = random_dihedral_matrix()
    return apply_affine_mat(x, mat)


def random_affine_combo(x: tf.Tensor,
                        max_rot_deg: float = 10.0, p_rot: float = 0.75,  # rotation
                        max_zoom: float = 1.1, row_pct: float = 0.5, col_pct: float = 0.5, p_zoom=0.75,  # zoom
                        max_shear_deg: float = 10, p_shear: float = 0.0,  # shear
                        wrg: float = 0.1, hrg: float = 0.1, p_shift=0.0,  # shift
                        ) -> tf.Tensor:
    mats = [random_rotate_matrix(max_rot_deg),
            random_zoom_matrix(max_zoom, row_pct, col_pct),
            random_shear_matrix(max_shear_deg),
            random_shift_matrix(wrg, hrg),
            ]
    ps = [p_rot,
          p_zoom,
          p_shear,
          p_shift,
          ]
    return apply_affine_mats(x, mats, ps)


def random_lighting(x: tf.Tensor, max_lighting: float = 0.2) -> tf.Tensor:
    x = tf.image.random_brightness(x, 0.5 * max_lighting)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = tf.image.random_contrast(x, 1 - max_lighting, 1 / (1 - max_lighting))
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


def tfm_random_brightness(max_delta: float) -> Callable:
    return functools.partial(tf.image.random_brightness, max_delta=max_delta)


def tfm_random_contrast(lower: float, upper: float) -> Callable:
    return functools.partial(tf.image.random_contrast, lower=lower, upper=upper)


def fastai_transforms(x: tf.Tensor,
                      do_flip: bool = True, flip_vert: bool = False,
                      max_rotate: float = 10., max_zoom: float = 1.1,
                      max_lighting: float = 0.2,
                      p_affine: float = 0.75,
                      p_lighting: float = 0.75,
                      ) -> tf.Tensor:
    mats = []
    ps = []

    if do_flip:
        if flip_vert:
            mats.append(random_dihedral_matrix())
            ps.append(1.0)
        else:
            mats.append(flip_matrix())
            ps.append(0.5)

    if max_rotate:
        mats.append(random_rotate_matrix(max_rotate))
        ps.append(p_affine)

    if max_zoom > 1:
        mats.append(random_zoom_matrix(max_zoom))
        ps.append(p_affine)

    x = apply_affine_mats(x, mats, ps)

    if max_lighting:
        x = core.random_transform(x, tfm_random_brightness(0.5 * max_lighting), p_lighting)
        x = core.random_transform(x, tfm_random_contrast(1 - max_lighting, 1 / (1 - max_lighting)), p_lighting)

    return tf.clip_by_value(x, 0.0, 1.0)


def random_pad_crop(x: tf.Tensor, pad_size: int) -> tf.Tensor:
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
    :param x: Input image of shape [h, w, 3] or a batch of images of shape [n, h, h, 3]. Each pixel must be already
              scaled to [0, 1].
    :return: Normalized image.
    """
    return (x - 0.5) * 2.0


def reverse_imagenet_normalize_tf(x: tf.Tensor) -> tf.Tensor:
    return x / 2.0 + 0.5


def imagenet_normalize_pytorch(x: tf.Tensor) -> tf.Tensor:
    """
    Default PyTorch image normalization for Keras models pre-trained on ImageNet.
    :param x: Input image of shape [h, w, 3] or a batch of images of shape [n, h, h, 3]. Each pixel must be already
              scaled to [0, 1].
    :return: Normalized image.
    """
    return (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


def reverse_imagenet_normalize_pytorch(x: tf.Tensor) -> tf.Tensor:
    return x * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]


def imagenet_normalize_caffe(x: tf.Tensor) -> tf.Tensor:
    """
    Default Caffe image normalization for Keras models pre-trained on ImageNet.
    :param x: Input image of shape [h, w, 3] or a batch of images of shape [n, h, h, 3]. Each pixel must be already
              scaled to [0, 1].
    :return: Normalized image.
    """
    return x[..., ::-1] * 255 - [103.939, 116.779, 123.68]


def reverse_imagenet_normalize_caffe(x: tf.Tensor) -> tf.Tensor:
    return ((x + [103.939, 116.779, 123.68]) / 255.0)[..., ::-1]


REVERSE_IMAGENET_NORMALIZE = {
    imagenet_normalize_tf: reverse_imagenet_normalize_tf,
    imagenet_normalize_pytorch: reverse_imagenet_normalize_pytorch,
    imagenet_normalize_caffe: reverse_imagenet_normalize_caffe,
}


def standard_scaler(x: tf.Tensor, mean, std) -> tf.Tensor:
    """
    Normalize an input image by subtracting a given mean and then divide by a given standard deviation.

    :param x: Input image.
    :param mean: Mean value of the pixels.
    :param std: Standard deviation of the pixels.
    :return: Normalized image.
    """
    return (x - mean) / std


def reverse_standard_scaler(x: tf.Tensor, mean, std) -> tf.Tensor:
    return x * std + mean


def set_shape(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    x.set_shape([h, w, c])
    return x


def tfm_set_shape(h: int = None, w: int = None, c: int = 3) -> Callable:
    return functools.partial(set_shape, h=h, w=w, c=c)


def tfm_resize(h: int, w: int) -> Callable:
    return functools.partial(tf.image.resize_images, size=[h, w])


def tfm_random_flip(flip_vert: bool = False) -> Callable:
    return functools.partial(random_flip, flip_vert=flip_vert)


def tfm_central_crop(center_frac: float = 1.0) -> Callable:
    return functools.partial(tf.image.central_crop, central_fraction=center_frac)


def tfm_pad_crop(pad_size: int) -> Callable:
    return functools.partial(random_pad_crop, pad_size=pad_size)


def tfm_standard_scaler(mean, std) -> Callable:
    return functools.partial(standard_scaler, mean=mean, std=std)


def tfm_reverse_standard_scaler(mean, std) -> Callable:
    return functools.partial(reverse_standard_scaler, mean=mean, std=std)


def tfm_cutout(h: int, w: int) -> Callable:
    return functools.partial(cutout, h=h, w=w)


def tfm_fastai(do_flip: bool = True, flip_vert: bool = False, max_rotate: float = 10., max_zoom: float = 1.1,
               max_lighting: float = 0.2, p_affine: float = 0.75, p_lighting: float = 0.75) -> Callable:
    return functools.partial(fastai_transforms, do_flip=do_flip, flip_vert=flip_vert, max_rotate=max_rotate,
                             max_zoom=max_zoom, max_lighting=max_lighting, p_affine=p_affine, p_lighting=p_lighting)


def get_inception_transforms(h: int, w: int, training: bool, flip_vert: bool = False, center_frac: float = 1.0,
                             normalizer=imagenet_normalize_tf) -> List[Callable]:
    """
    Sequence of transforms as used in Google's InceptionV3 code.

    :param h: Target height of a data image, e.g., 299 in Inception models.
    :param w: Target width of a data image, e.g., 299 in Inception models.
    :param training: Whether the transforms are applied to a training set.
    :param flip_vert: Whether to perform random vertical flipping in the training input pipeline.
    :param center_frac: Fraction of center crop. Applied to evaluation and not to training.
    :param normalizer: Data normalization function. Default to Tensorflow's ImageNet noramlization function, i.e.,
    `x = (x-0.5)*2`.
    :return: A function that parses a TFExample into an image.
    """

    if training:
        return [distorted_bbox_crop, tfm_set_shape(), tfm_resize(h, w), tfm_random_flip(flip_vert), distort_color,
                normalizer]
    return [tfm_central_crop(center_frac), tfm_set_shape(), tfm_resize(h, w), normalizer]


def get_fastai_transforms(h: int, w: int, training: bool, normalizer=imagenet_normalize_tf) -> List[Callable]:
    if training:
        return [tfm_set_shape(), tfm_resize(h, w), tfm_fastai(), normalizer]
    return [tfm_set_shape(), tfm_resize(h, w), normalizer]
