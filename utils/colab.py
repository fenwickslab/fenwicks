from google import colab
import tensorflow as tf
import os
import json

TPU_ADDRESS = f'grpc://{os.environ["COLAB_TPU_ADDR"]}' if "COLAB_TPU_ADDR" in os.environ else None


def setup_gcs():
    colab.auth.authenticate_user()
    with tf.Session(TPU_ADDRESS) as sess:
        with open('/content/adc.json', 'r') as f:
            auth_info = json.load(f)
        tf.contrib.cloud.configure_gcs(sess, credentials=auth_info)


def upload_files():
    """
    Upload one or more files from physical computer to Colab's virtual machine.
    :return: None.
    """
    uploaded = colab.files.upload()
    for fn in uploaded.keys():
        tf.logging.info(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')


def download_file(fn: str):
    """
    Download a file from Colab's virtual machine to physical computer.

    :param fn: file name on Colab
    :return: None.
    """
    colab.files.download(fn)


def mount_google_drive():
    colab.drive.mount('/content/gdrive')
