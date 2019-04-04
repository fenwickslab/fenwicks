from ...io import *
from typing import List, Tuple
from keras_applications import vgg16, resnet50, inception_resnet_v2, inception_v3, xception, mobilenet_v2, nasnet, \
    resnext


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


def VGG16_weights(model_dir: str, include_top: bool = False, overwrite: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(vgg16.VGG16, model_dir=model_dir, include_top=include_top)


def VGG16_size() -> int:
    return 224


def get_VGG16(pooling: str = None):
    return vgg16.VGG16(include_top=False, weights=None, pooling=pooling)


def ResNet50_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(resnet50.ResNet50, model_dir=model_dir, include_top=include_top)


def ResNet50_size() -> int:
    return 224


def get_ResNet50(pooling: str = None):
    return resnet50.ResNet50(include_top=False, weights=None, pooling=pooling)


def ResNeXt50_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(resnext.ResNeXt50, model_dir=model_dir, include_top=include_top)


def ResNeXt50_size() -> int:
    return 224


def get_ResNeXt50(pooling: str = None):
    return resnext.ResNeXt50(include_top=False, weights=None, pooling=pooling)


def InceptionV3_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(inception_v3.InceptionV3, model_dir=model_dir, include_top=include_top)


def InceptionV3_size() -> int:
    return 299


def get_InceptionV3(pooling: str = None):
    return inception_v3.InceptionV3(include_top=False, weights=None, pooling=pooling)


def InceptionResNetV2_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(inception_resnet_v2.InceptionResNetV2, model_dir=model_dir, include_top=include_top)


def InceptionResNetV2_size() -> int:
    return 299


def get_InceptionResNetV2(pooling: str = None):
    return inception_resnet_v2.InceptionResNetV2(include_top=False, weights=None, pooling=pooling)


def MobileNetV2_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(mobilenet_v2.MobileNetV2, model_dir=model_dir, include_top=include_top)


def MobileNetV2_size() -> int:
    return 224


def get_MobileNetV2(pooling: str = None):
    return mobilenet_v2.MobileNetV2(include_top=False, weights=None, pooling=pooling)


def Xception_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(xception.Xception, model_dir=model_dir, include_top=include_top)


def Xception_size() -> int:
    return 299


def get_Xception(pooling: str = None):
    return xception.Xception(include_top=False, weights=None, pooling=pooling)


def NASNetLarge_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(nasnet.NASNetLarge, model_dir=model_dir, include_top=include_top)


def NASNetLarge_size() -> int:
    return 331


def get_NASNetLarge(pooling: str = None):
    return nasnet.NASNetLarge(include_top=False, weights=None, pooling=pooling)


def freeze(model):
    for l in model.layers:
        l.trainable = False


def unfreeze(model):
    for l in model.layers:
        l.trainable = True
