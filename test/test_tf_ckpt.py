import tensorflow as tf
import fenwicks as fw


def setup_one_var_ckpt(model_name: str = 'my-model', global_step: int = 1):
    v = tf.get_variable("v", [1])
    op = tf.assign(v, [12345])

    with tf.Session() as sess:
        saver = tf.train.Saver([v])
        sess.run(op)
        saver.save(sess, model_name, global_step=global_step)


def remove_ckpt(model_name: str, global_step: int):
    ckpt_fn = f'{model_name}-{global_step}.*'
    fw.io.remove_file_pattern(ckpt_fn)
    tf.io.gfile.remove('checkpoint')


def test_ckpt_saver():
    setup_one_var_ckpt()

    assert tf.io.gfile.exists('my-model-1.index')
    assert tf.io.gfile.exists('my-model-1.meta')
    assert tf.io.gfile.exists('my-model-1.data-00000-of-00001')
    assert tf.io.gfile.exists('checkpoint')

    remove_ckpt('my-model', 1)


def test_ckpt_var_shape_dtype():
    setup_one_var_ckpt()
    global_step = 1
    ckpt_fn = f'my-model-{global_step}'

    reader = tf.train.NewCheckpointReader(ckpt_fn)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    assert var_to_shape_map == {'v': [1]}
    assert var_to_dtype_map == {'v': tf.float32}

    remove_ckpt('my-model', 1)


def test_ckpt_read():
    setup_one_var_ckpt()
    global_step = 1
    ckpt_fn = f'my-model-{global_step}'

    new_v = tf.get_variable("new_v", [1])
    with tf.Session() as sess:
        tf.train.init_from_checkpoint(ckpt_fn, {'v': 'new_v'})
        init_op = tf.initializers.global_variables()
        sess.run(init_op)
        result = sess.run(new_v)
        assert result == 12345

    remove_ckpt('my-model', 1)


def test_keras_ckpt_vgg16():
    fw.keras_models.get_model('VGG16', root_dir='.')
    tf.io.gfile.rmtree('./model')
