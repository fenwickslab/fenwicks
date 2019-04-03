from ...io import *
from typing import List, Tuple


def get_ws_vars(ws_ckpt_fn: str) -> List[str]:
    reader = tf.train.NewCheckpointReader(ws_ckpt_fn)
    var_to_shape_map = reader.get_variable_to_shape_map()
    ws_vars = list(var_to_shape_map.keys())

    n = len(ws_vars)
    for i in range(n):
        ws_vars[i] = ws_vars[i] + '[^/]'

    return ws_vars


def keras_model_weights(model_class, model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MkDir(model_dir)
        model = model_class(include_top=include_top)
        # Here we use the simplest SGD optimizer to avoid creating new variables
        model.compile(tf.train.GradientDescentOptimizer(0.1), 'categorical_crossentropy')
        tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)

    ws_dir = os.path.join(model_dir, 'keras')
    ws_ckpt_fn = os.path.join(ws_dir, 'keras_model.ckpt')
    ws_vars = get_ws_vars(ws_ckpt_fn)
    return ws_dir, ws_vars


def VGG16_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.VGG16, model_dir=model_dir, include_top=include_top)


def ResNet50_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.ResNet50, model_dir=model_dir, include_top=include_top)


def InceptionV3_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.InceptionV3, model_dir=model_dir, include_top=include_top)


def InceptionResNetV2_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.InceptionResNetV2, model_dir=model_dir, include_top=include_top)


def MobileNetV2_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.MobileNetV2, model_dir=model_dir, include_top=include_top)


def Xception_weights(model_dir: str, include_top: bool = False) -> Tuple[str, List[str]]:
    return keras_model_weights(tf.keras.applications.Xception, model_dir=model_dir,
                               include_top=include_top)


def get_VGG16(pooling: str = None):
    return tf.keras.applications.VGG16(include_top=False, weights=None, pooling=pooling)


def get_ResNet50(pooling: str = None):
    return tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=pooling)


def get_InceptionV3(pooling: str = None):
    return tf.keras.applications.InceptionV3(include_top=False, weights=None, pooling=pooling)


def get_InceptionResNetV2(pooling: str = 'avg'):
    return tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, pooling=pooling)


def get_MobileNetV2(pooling: str = None):
    return tf.keras.applications.MobileNetV2(include_top=False, weights=None, pooling=pooling)


def get_Xception(pooling: str = None):
    return tf.keras.applications.Xception(include_top=False, weights=None, pooling=pooling)


def freeze(model):
    for l in model.layers:
        l.trainable = False


def unfreeze(model):
    for l in model.layers:
        l.trainable = True
