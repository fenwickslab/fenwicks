import tensorflow as tf
import fenwicks as fw


def test_ckpt_saver():
    v = tf.get_variable("v", [1])
    op = tf.assign(v, [12345])
    global_step = 1

    with tf.Session() as sess:
        saver = tf.train.Saver([v])
        sess.run(op)
        saver.save(sess, 'my-model', global_step=global_step)

    assert tf.io.gfile.exists('my-model-1.index')
    assert tf.io.gfile.exists('my-model-1.meta')
    assert tf.io.gfile.exists('my-model-1.data-00000-of-00001')
    assert tf.io.gfile.exists('checkpoint')

    fw.io.remove_file_pattern('my-model-1.*')
    tf.io.gfile.remove('checkpoint')
