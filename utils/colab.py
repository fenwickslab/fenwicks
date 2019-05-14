from ..io import create_clean_dir

from google import colab
import tensorflow as tf
import os
import json

TPU_ADDRESS = f'grpc://{os.environ["COLAB_TPU_ADDR"]}' if "COLAB_TPU_ADDR" in os.environ else None


def setup_gcs(tpu_address: str = None):
    """
    Set up Google Cloud Storage for TPU.
    :param tpu_address: network address of the TPU, starting with 'grpc://'. Default: Colab's TPU address.
    :return: None
    """
    colab.auth.authenticate_user()

    tpu_address = tpu_address or TPU_ADDRESS

    with tf.Session(tpu_address) as sess:
        with open('/content/adc.json', 'r') as f:
            auth_info = json.load(f)
        tf.contrib.cloud.configure_gcs(sess, credentials=auth_info)


def upload_files():
    """
    Upload one or more files from physical computer to Colab's virtual machine.
    :return: None.
    """
    colab.files.upload()


def download_file(fn: str):
    """
    Download a file from Colab's virtual machine to physical computer.

    :param fn: file name on Colab
    :return: None.
    """
    colab.files.download(fn)


def mount_google_drive(gdrive_path: str = './gdrive'):
    """
    Mount Google Drive. Do nothing if Google Drive is already mounted.
    :param gdrive_path: local path to mount Google Drive to.
    :return: None.
    """
    colab.drive.mount(gdrive_path)


def setup_kaggle_from_gdrive(gdrive_path: str = './gdrive/My Drive/kaggle.json',
                             local_path: str = '/root/.kaggle/kaggle.json'):
    if not tf.io.gfile.exists(local_path):
        mount_google_drive()
        create_clean_dir(os.path.dirname(local_path))
        tf.io.gfile.copy(gdrive_path, local_path)
        os.chmod(local_path, 600)
    else:
        tf.logging.info(f'Kaggle already set up. Skipping.')
