from ...io import *
from ..transform import imagenet_normalize_tf, imagenet_normalize_caffe, imagenet_normalize_pytorch
from typing import List, Tuple
from keras_applications import vgg16, resnet50, resnet_v2, inception_resnet_v2, inception_v3, xception, mobilenet_v2, \
    nasnet, resnext


def get_ws_vars(ws_ckpt_fn: str) -> List[str]:
    reader = tf.train.NewCheckpointReader(ws_ckpt_fn)
    var_to_shape_map = reader.get_variable_to_shape_map()
    ws_vars = list(var_to_shape_map.keys())

    n = len(ws_vars)
    for i in range(n):
        ws_vars[i] = ws_vars[i] + '[^/]'

    return ws_vars


def keras_model_weights(model_class, model_dir: str, include_top: bool = False, overwrite: bool = False) -> Tuple[
    str, List[str]]:
    if overwrite or (not tf.gfile.Exists(model_dir)):
        create_clean_dir(model_dir)
        model = model_class(include_top=include_top)
        # Here we use the simplest SGD optimizer to avoid creating new variables
        model.compile(tf.train.GradientDescentOptimizer(0.1), 'categorical_crossentropy')
        tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)

    ws_dir = os.path.join(model_dir, 'keras')
    ws_ckpt_fn = os.path.join(ws_dir, 'keras_model.ckpt')
    ws_vars = get_ws_vars(ws_ckpt_fn)
    return ws_dir, ws_vars


def get_vgg16(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 224
    model = vgg16.VGG16(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(vgg16.VGG16, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_caffe
    return model, weight_dir, weight_vars, img_size, normalizer


def get_resnet50(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 224
    model = resnet50.ResNet50(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(resnet50.ResNet50, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_caffe
    return model, weight_dir, weight_vars, img_size, normalizer


def get_resnet50_v2(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 224
    model = resnet_v2.ResNet50V2(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(resnet_v2.ResNet50V2, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def get_resnext50(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 224
    model = resnext.ResNeXt50(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(resnext.ResNeXt50, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_pytorch
    return model, weight_dir, weight_vars, img_size, normalizer


def get_inception_v3(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 299
    model = inception_v3.InceptionV3(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(inception_v3.InceptionV3, model_dir=model_dir,
                                                  include_top=include_top, overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def InceptionResNetV2_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(inception_resnet_v2.InceptionResNetV2, model_dir=model_dir, include_top=include_top)


def get_inception_resnet_v2(model_dir: str, include_top: bool = False, pooling: str = None,
                            overwrite: bool = False) -> Tuple:
    img_size = 299
    model = inception_resnet_v2.InceptionResNetV2(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(inception_resnet_v2.InceptionResNetV2, model_dir=model_dir,
                                                  include_top=include_top, overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def get_mobilenet_v2(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 224
    model = mobilenet_v2.MobileNetV2(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(mobilenet_v2.MobileNetV2, model_dir=model_dir,
                                                  include_top=include_top, overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def get_xception(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 299
    model = xception.Xception(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(xception.Xception, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def get_nasnet_large(model_dir: str, include_top: bool = False, pooling: str = None, overwrite: bool = False) -> Tuple:
    img_size = 331
    model = nasnet.NASNetLarge(include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(nasnet.NASNetLarge, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    normalizer = imagenet_normalize_tf
    return model, weight_dir, weight_vars, img_size, normalizer


def freeze(model):
    model.trainable = False
    for l in model.layers:
        l.trainable = False


def unfreeze(model):
    model.trainable = True
    for l in model.layers:
        l.trainable = True
