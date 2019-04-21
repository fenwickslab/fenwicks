import tensorflow as tf
import numpy as np
from typing import Tuple
from .. import data


def compute_image_mean_std(fn_data: str, n_data: int, batch_size: int = 1) -> Tuple[float, float, int, int]:
    ds = data.tfrecord_ds(fn_data, data.tfexample_image_parser, batch_size=batch_size, training=False)
    data_op = ds.make_one_shot_iterator().get_next()

    x_count, x_sum, x_sum_sq = 0.0, np.zeros(3), np.zeros(3)

    with tf.Session() as sess:
        for _ in range(n_data // batch_size):
            x, _ = sess.run(data_op)
            x = x.astype(np.float64)
            x_count += np.prod(x.shape[:3])
            x_sum += np.sum(x, axis=(0, 1, 2))
            x_sum_sq += np.sum(x * x, axis=(0, 1, 2))

    h, w = x.shape[1], x.shape[2]

    x_mean = x_sum / x_count
    x_std = np.sqrt((x_count * x_sum_sq - x_sum * x_sum) / (x_count * (x_count - 1)))

    return x_mean, x_std, h, w
