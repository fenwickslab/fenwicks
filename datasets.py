import tensorflow as tf
import os
from urllib.parse import urlparse

__all__ = ['URLs', 'untar_data']

class URLs():
  FASTAI = 'http://files.fast.ai/data/'
  DVC = f'{FASTAI}dogscats.zip'
  CIFAR10 = f'{FASTAI}cifar10.tgz'

  TF = 'http://download.tensorflow.org/data/'
  SPEECH_CMD_001 = f'{TF}speech_commands_v0.01.tar.gz'
  SPEECH_CMD_002 = f'{TF}speech_commands_v0.02.tar.gz'

def untar_data(url:str, dest:str='.')->str:
  if not os.path.isdir(dest):
    os.mkdir(dest)
  url_path = urlparse(url).path
  fn = os.path.basename(url_path)
  tf.keras.utils.get_file(fn, origin=url, extract=True, cache_dir=dest)
  return os.path.join(dest, 'datasets')
