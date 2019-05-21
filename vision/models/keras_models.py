from ...imports import *

from ...io import get_model_dir, create_clean_dir
from ..transform import imagenet_normalize_tf, imagenet_normalize_caffe, imagenet_normalize_pytorch

from collections import namedtuple
from keras_applications import vgg16, vgg19, resnet, resnet50, resnext, resnet_v2, inception_resnet_v2, inception_v3, \
    xception, mobilenet, mobilenet_v2, nasnet, densenet


def get_ws_vars(ws_ckpt_fn: str) -> List[str]:
    reader = tf.train.NewCheckpointReader(ws_ckpt_fn)
    var_to_shape_map = reader.get_variable_to_shape_map()
    ws_vars = list(var_to_shape_map.keys())

    n = len(ws_vars)
    for i in range(n):
        ws_vars[i] = ws_vars[i] + '[^/]'

    return ws_vars


def keras_ckpt(model, model_dir: str):
    # Here we use the simplest SGD optimizer to avoid creating new variables
    model.compile(tf.train.GradientDescentOptimizer(0.1), 'categorical_crossentropy')
    tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)


def keras_model_weights(model_class: Callable, model_dir: str, include_top: bool = False, overwrite: bool = False) -> \
        Tuple[str, List[str]]:
    if overwrite or (not tf.io.gfile.exists(model_dir)):
        create_clean_dir(model_dir)
        model = model_class(include_top=include_top)
        keras_ckpt(model, model_dir)

    ws_dir = os.path.join(model_dir, 'keras')
    ws_ckpt_fn = os.path.join(ws_dir, 'keras_model.ckpt')
    ws_vars = get_ws_vars(ws_ckpt_fn)
    return ws_dir, ws_vars


KerasModel = namedtuple('KerasModel', ['model_func', 'weight_dir', 'weight_vars', 'img_size', 'normalizer'])


def get_keras_model(keras_model, img_size, normalizer, model_dir: str, include_top: bool = False, pooling: str = None,
                    overwrite: bool = False) -> KerasModel:
    model_func = functools.partial(keras_model, include_top=include_top, weights=None, pooling=pooling)
    weight_dir, weight_vars = keras_model_weights(keras_model, model_dir=model_dir, include_top=include_top,
                                                  overwrite=overwrite)
    keras_model = KerasModel(model_func, weight_dir, weight_vars, img_size, normalizer)
    return keras_model


def get_model(model_name: str, bucket: str, model_dir: str = None, include_top: bool = False, pooling: str = None,
              overwrite: bool = False) -> KerasModel:
    models = {
        'VGG16': dict(keras_model=vgg16.VGG16, img_size=224, normalizer=imagenet_normalize_caffe),
        'VGG19': dict(keras_model=vgg19.VGG19, img_size=224, normalizer=imagenet_normalize_caffe),

        'ResNet50': dict(keras_model=resnet50.ResNet50, img_size=224, normalizer=imagenet_normalize_caffe),
        'ResNet101': dict(keras_model=resnet.ResNet101, img_size=224, normalizer=imagenet_normalize_caffe),
        'ResNet152': dict(keras_model=resnet.ResNet152, img_size=224, normalizer=imagenet_normalize_caffe),

        # Note: ResNetV2 image size is 224, not 299 as stated on tf.slim web site
        'ResNet50V2': dict(keras_model=resnet_v2.ResNet50V2, img_size=224, normalizer=imagenet_normalize_tf),
        'ResNet101V2': dict(keras_model=resnet_v2.ResNet101V2, img_size=224, normalizer=imagenet_normalize_tf),
        'ResNet152V2': dict(keras_model=resnet_v2.ResNet152V2, img_size=224, normalizer=imagenet_normalize_tf),

        'InceptionV3': dict(keras_model=inception_v3.InceptionV3, img_size=299, normalizer=imagenet_normalize_tf),
        'InceptionResNetV2': dict(keras_model=inception_resnet_v2.InceptionResNetV2, img_size=299,
                                  normalizer=imagenet_normalize_tf),
        'Xception': dict(keras_model=xception.Xception, img_size=299, normalizer=imagenet_normalize_tf),

        # todo: alpha parameter for mobilenet
        'MobileNet': dict(keras_model=mobilenet.MobileNet, img_size=224, normalizer=imagenet_normalize_tf),
        'MobileNetV2': dict(keras_model=mobilenet_v2.MobileNetV2, img_size=224, normalizer=imagenet_normalize_tf),

        # fixme
        'ResNeXt50': dict(keras_model=resnext.ResNeXt50, img_size=224, normalizer=imagenet_normalize_pytorch),
        'ResNeXt101': dict(keras_model=resnext.ResNeXt101, img_size=224, normalizer=imagenet_normalize_pytorch),

        # fixme
        'NASNetLarge': dict(keras_model=nasnet.NASNetLarge, img_size=331, normalizer=imagenet_normalize_tf),
        'NASNetMobile': dict(keras_model=nasnet.NASNetMobile, img_size=224, normalizer=imagenet_normalize_tf),

        'DenseNet121': dict(keras_model=densenet.DenseNet121, img_size=224, normalizer=imagenet_normalize_pytorch),
        'DenseNet169': dict(keras_model=densenet.DenseNet169, img_size=224, normalizer=imagenet_normalize_pytorch),
        'DenseNet201': dict(keras_model=densenet.DenseNet201, img_size=224, normalizer=imagenet_normalize_pytorch),
    }

    model_dir = model_dir or get_model_dir(bucket, model_name)
    params = dict(**models[model_name], model_dir=model_dir, include_top=include_top, pooling=pooling,
                  overwrite=overwrite)
    return get_keras_model(**params)
