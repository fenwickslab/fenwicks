import fenwicks as fw
import tensorflow as tf
import numpy as np


def test_set_random_seed():
    fw.core.set_random_seed()
    op = tf.random.uniform([2, 2])

    with tf.Session() as sess:
        x = sess.run(op)

    assert np.allclose(x, [[0.06672299, 0.3009504], [0.85697794, 0.1031245]])
