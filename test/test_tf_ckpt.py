import tensorflow as tf
import fenwicks as fw
import numpy as np

from typing import Dict


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


def test_keras_vgg16_ckpt():
    fw.keras_models.get_model('VGG16', root_dir='.')
    tf.io.gfile.rmtree('./model')


def count_vars(var_to_shape_map: Dict):
    n_weight = 0
    n_var = 0
    for k, v in var_to_shape_map.items():
        if k != 'global_step':
            # print(k, ':', v, ',', np.prod(v))
            n_weight += np.prod(v)
        n_var += 1
    return n_weight, n_var


def check_keras_model_num_vars(model_name: str, expected_n_weight: int = -1):
    fw.keras_models.get_model(model_name, root_dir='.')

    ckpt_fn = f'./model/{model_name}/keras/keras_model.ckpt'
    reader = tf.train.NewCheckpointReader(ckpt_fn)
    var_to_shape_map = reader.get_variable_to_shape_map()
    n_weight, n_var = count_vars(var_to_shape_map)
    if expected_n_weight > 0:
        assert n_weight == expected_n_weight
    else:
        print('Total number of variables in checkpoint:', n_weight)

    tf.io.gfile.rmtree('./model')


def test_keras_vgg16_num_vars():
    check_keras_model_num_vars('VGG16', 14714688)


def test_keras_vgg19_num_vars():
    check_keras_model_num_vars('VGG19', 20024384)


def test_keras_resnet50_num_vars():
    check_keras_model_num_vars('ResNet50', 23587712)


def test_keras_resnet101_num_vars():
    check_keras_model_num_vars('ResNet101', 42658176)


def test_keras_resnet152_num_vars():
    check_keras_model_num_vars('ResNet152', 58370944)


def test_keras_resnet50v2_num_vars():
    check_keras_model_num_vars('ResNet50V2', 23564800)


def test_keras_resnet101v2_num_vars():
    check_keras_model_num_vars('ResNet101V2', 42626560)


def test_keras_resnet152v2_num_vars():
    check_keras_model_num_vars('ResNet152V2', 58331648)


# fixme
def test_keras_resnext50_num_vars():
    check_keras_model_num_vars('ResNeXt50')


def test_keras_inceptionv3_num_vars():
    check_keras_model_num_vars('InceptionV3', 21802784)


def test_keras_inceptionresnetv2_num_vars():
    check_keras_model_num_vars('InceptionResNetV2', 54336736)


def test_keras_xception_num_vars():
    check_keras_model_num_vars('Xception', 20861480)


# todo: mobilenet

def test_keras_densenet121_num_vars():
    check_keras_model_num_vars('DenseNet121', 7037504)


def test_keras_densenet169_num_vars():
    check_keras_model_num_vars('DenseNet169', 12642880)


def test_keras_densenet201_num_vars():
    check_keras_model_num_vars('DenseNet201', 18321984)


def test_keras_nasnetlarge_num_vars():
    check_keras_model_num_vars('NASNetLarge', 84916818)


def test_keras_nasnetmobile_num_vars():
    check_keras_model_num_vars('NASNetMobile', 4269716)
