import tensorflow as tf
import os
from fenwicks.io import *

def keras_model_ckpt(model_class, model_dir:str, include_top:bool=False):
  create_clean_dir(model_dir)
  model = model_class(include_top=include_top)
  model.compile(tf.train.GradientDescentOptimizer(0.1),
    'categorical_crossentropy')
  est = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)

def InceptionResNetV2_ckpt(model_dir:str, include_top:bool=False):
  keras_model_ckpt(tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    model_dir=model_dir, include_top=include_top)
