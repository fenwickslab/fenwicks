import tensorflow as tf


def get_pixel_value(img: tf.Tensor, x, y) -> tf.Tensor:
    x_shape = tf.shape(x)
    B = x_shape[0]
    H = x_shape[1]
    W = x_shape[2]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, H, W))

    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)


def reflect(x, max_x):
    x = tf.abs(x)
    x = max_x - tf.abs(max_x - x)
    return x


def bilinear_sampler(img: tf.Tensor, x, y, do_reflect: bool = True) -> tf.Tensor:
    print(do_reflect)
    img_shape = tf.shape(img)
    H = img_shape[1]
    W = img_shape[2]

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    if do_reflect:
        x0r = reflect(x0, max_x)
        x1r = reflect(x1, max_x)
        y0r = reflect(y0, max_y)
        y1r = reflect(y1, max_y)
    else:
        x0r = tf.clip_by_value(x0, zero, max_x)
        x1r = tf.clip_by_value(x1, zero, max_x)
        y0r = tf.clip_by_value(y0, zero, max_y)
        y1r = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0r, y0r)
    Ib = get_pixel_value(img, x0r, y1r)
    Ic = get_pixel_value(img, x1r, y0r)
    Id = get_pixel_value(img, x1r, y1r)

    if do_reflect:
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)
    else:
        x0 = tf.cast(x0r, tf.float32)
        x1 = tf.cast(x1r, tf.float32)
        y0 = tf.cast(y0r, tf.float32)
        y1 = tf.cast(y1r, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


def affine_grid_generator(H: int, W: int, tfm_mat) -> tf.Tensor:
    B = tf.shape(tfm_mat)[0]

    x = tf.linspace(-1.0, 1.0, W)
    y = tf.linspace(-1.0, 1.0, H)
    x_t, y_t = tf.meshgrid(x, y)

    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid B times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([B, 1, 1]))

    # cast to float32 (required for matmul)
    tfm_mat = tf.cast(tfm_mat, tf.float32)
    sampling_grid = tf.cast(sampling_grid, tf.float32)

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(tfm_mat, sampling_grid)
    # batch grid has shape (B, 2, H*W)

    # reshape to (B, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [B, 2, H, W])

    return batch_grids


def affine_transform(X: tf.Tensor, tfm_mat, out_dims=None, do_reflect: bool = True) -> tf.Tensor:
    X_shape = tf.shape(X)
    B = X_shape[0]
    H = X_shape[1]
    W = X_shape[2]
    tfm_mat = tf.reshape(tfm_mat, [B, 2, 3])

    out_H, out_W = out_dims if out_dims else H, W
    batch_grids = affine_grid_generator(out_H, out_W, tfm_mat)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    return bilinear_sampler(X, x_s, y_s, do_reflect)
