import tensorflow as tf
import os

def keras_model_ckpt(model_class, model_dir:str, include_top:bool=False):
  create_clean_dir(model_dir)
  model = model_class(include_top=include_top)
  model.compile(tf.train.AdamOptimizer(), 'categorical_crossentropy')
  est = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)

  for f in tf.gfile.Glob(os.path.join(model_dir, 'keras/*')):
    tf.gfile.Rename(f, os.path.join(model_dir, os.path.basename(f)))

  tf.gfile.DeleteRecursively(os.path.join(model_dir, 'keras'))

def InceptionResNetV2_ckpt(model_dir:str, include_top:bool=False):
  keras_model_ckpt(tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    model_dir=model_dir, include_top=include_top)
