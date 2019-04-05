import tensorflow as tf


def distort_color(x, cb_distortion_range=0.1, cr_distortion_range=0.1):
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


def distorted_bbox_crop(x, min_object_covered=0.1, aspect_ratio_range=(3. / 4., 4. / 3.),
                        area_range=(0.05, 1.0), max_attempts=100):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_sz, _ = tf.image.sample_distorted_bounding_box(tf.shape(x), bounding_boxes=bbox,
                                                                    min_object_covered=min_object_covered,
                                                                    aspect_ratio_range=aspect_ratio_range,
                                                                    area_range=area_range,
                                                                    max_attempts=max_attempts,
                                                                    use_image_if_no_bounding_boxes=True)
    x = tf.slice(x, bbox_begin, bbox_sz)
    return x
