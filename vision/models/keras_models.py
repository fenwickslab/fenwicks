import tensorflow as tf
import os
from fenwicks.io import *
from typing import List, Tuple

def keras_model_ckpt(model_class, model_dir:str,
  include_top:bool=False)->Tuple[str, str]:

  create_clean_dir(model_dir)
  model = model_class(include_top=include_top)
  model.compile(tf.train.GradientDescentOptimizer(0.1),
    'categorical_crossentropy')
  tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)

  ws_dir = os.path.join(model_dir, 'keras')
  ws_ckpt_fn = os.path.join(ws_dir, 'keras_model.ckpt')
  return ws_dir, ws_ckpt_fn

def InceptionResNetV2_ckpt(model_dir:str,
  include_top:bool=False)->Tuple[str, str]:

  return keras_model_ckpt(
    tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    model_dir=model_dir, include_top=include_top)

def get_ws_vars(ws_ckpt_fn:str)->List[str]:
  reader = tf.train.NewCheckpointReader(ws_ckpt_fn)
  var_to_shape_map = reader.get_variable_to_shape_map()
  ws_vars = list(var_to_shape_map.keys())

  n = len(ws_vars)
  for i in range(n):
    ws_vars[i] = ws_vars[i] + '[^/]'

  return ws_vars
