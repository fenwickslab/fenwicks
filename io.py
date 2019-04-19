from .vision.transform import *
from .core import apply_transforms

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import threading
import os
import itertools
import functools

from typing import List, Tuple
from tqdm import tqdm_notebook as tqdm
from tensorflow.contrib.tpu.python.tpu import datasets as tpu_datasets


def enum_files(data_dir: str, file_ext: str = 'jpg') -> List[str]:
    """
    Enumerate all files with a given extension in a given data directory.

    :param data_dir: Data directory.
    :param file_ext: Extensions of files to enumerate. Default: 'jpg'.
    :return: A list of file names. Note that these are base file names, not full paths.
    """
    file_pattern = os.path.join(data_dir, f'*.{file_ext}')
    matching_files = tf.gfile.Glob(file_pattern)
    return matching_files


def shuffle_paths_labels(paths: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    c = list(zip(paths, labels))
    random.shuffle(c)
    paths, labels = zip(*c)
    return list(paths), list(labels)


def find_files(data_dir: str, labels: List[str], shuffle=False, file_ext: str = 'jpg') -> Tuple[List[str], List[int]]:
    """
    Find all input files wth a given extension in specific subdirectories of a given data directory. Optionally shuffle
    the found file names.

    :param data_dir: Data directory.
    :param labels: Set of subdirectories (named after labels) in which to find files.
    :param shuffle: Whether to shuffle the outputs.
    :param file_ext: File extension. ONly find files with this extension.
    :return: Two lists: one for file paths and the other for their corresponding labels represented as indexes.
    """
    filepaths = []
    filelabels = []

    for i, label in enumerate(labels):
        matching_files = enum_files(os.path.join(data_dir, label), file_ext)
        filepaths.extend(matching_files)
        filelabels.extend([i] * len(matching_files))

    if shuffle:
        filepaths, filelabels = shuffle_paths_labels(filepaths, filelabels)

    return filepaths, filelabels


def find_files_with_label_csv(data_dir: str, csv_fn: str, shuffle=False, file_ext: str = 'jpg') -> Tuple[
    List[str], List[int], List[str]]:
    train_labels = pd.read_csv(csv_fn)
    labels = train_labels.label.unique()
    key_id = dict([(ee[1], ee[0]) for ee in enumerate(labels)])

    filepaths = []
    filelabels = []

    for _, row in train_labels.iterrows():
        filepaths.append(os.path.join(data_dir, f'{row["id"]}.{file_ext}'))
        filelabels.append(key_id[row['label']])

    if shuffle:
        filepaths, filelabels = shuffle_paths_labels(filepaths, filelabels)

    return filepaths, filelabels, labels


def create_clean_dir(path: str):
    """
    Create a new directory specified by `path`. If this directory already exists, delete all its files and
    subdirectories.

    :param path: Path to the directory to be created or cleaned.
    :return: None
    """
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.io.gfile.makedirs(path)


def file_size(fn: str) -> int:
    """
    Get the size of a file in bytes. Works for files on Google Cloud Storage.
    :param fn: Path to the file.
    :return: Size of the file.
    """
    stat = tf.io.gfile.stat(fn)
    return stat.length


def unzip(fn: str):
    """
    Extract a .zip or .7z file.

    :param fn: Name of the file to be decompressed.
    :param report_freq: Frequency of progress reporting.
    :return: None.
    """

    try:
        import libarchive.public
    except ImportError:
        raise ImportError('libarchive not installed. Run !apt install libarchive-dev and then !pip install libarchive.')

    for _ in tqdm(libarchive.public.file_pour(fn)):
        pass


def sub_dirs(data_dir: str, exclude_dirs: List[str] = []) -> List[str]:
    return [path for path in tf.gfile.ListDirectory(data_dir)
            if tf.gfile.IsDirectory(os.path.join(data_dir, path)) and path not in exclude_dirs]


def float_tffeature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_tffeature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_tffeature(value):
    if isinstance(value, str):
        value = bytes(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def raw_image_tfexample(raw_image, y):
    feat_dict = {'image': bytes_tffeature(raw_image),
                 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def numpy_tfexample(x, y):
    feat_dict = {'image': float_tffeature(x.tolist()),
                 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def numpy_tfrecord(X, y, output_file: str, overwrite: bool = False):
    n = X.shape[0]
    X_reshape = X.reshape(n, -1)

    if overwrite or not tf.gfile.Exists(output_file):
        with tf.io.TFRecordWriter(output_file) as record_writer:
            for i in tqdm(range(n)):
                example = numpy_tfexample(X_reshape[i], y[i])
                record_writer.write(example.SerializeToString())


def numpy_tfrecord_shards(x, y, output_file: str, num_shards: int = 2):
    spacing = np.linspace(0, len(x), num_shards + 1).astype(np.int)
    ranges = [[spacing[i], spacing[i + 1]] for i in range(num_shards)]

    threads = []
    for i in range(num_shards):
        start, end = ranges[i][0], ranges[i][1]
        args = (x[start:end], y[start:end],
                f'{output_file}-{i:05d}-of-{num_shards:05d}')
        t = threading.Thread(target=numpy_tfrecord, args=args)
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)


def files_tfrecord(paths: List[str], y: List[int], output_file: str, overwrite: bool = False, extractor=None):
    if overwrite or not tf.gfile.Exists(output_file):
        with tf.io.TFRecordWriter(output_file) as record_writer:
            for i, path in enumerate(tqdm(paths)):
                if extractor is None:
                    with tf.gfile.GFile(path, 'rb') as f:
                        img = f.read()
                        example = raw_image_tfexample(img, y[i])
                else:
                    img = extractor(path)
                    img = img.reshape(-1)
                    example = numpy_tfexample(img, y[i])
                record_writer.write(example.SerializeToString())


def data_dir_tfrecord(data_dir: str, output_file: str, shuffle: bool = False, overwrite: bool = False, extractor=None,
                      file_ext: str = 'jpg', exclude_dirs: List[str] = []) -> Tuple[List[str], List[int], List[str]]:
    """
    Create a TFRecords data file from the contents of a data directory `data_dir`. Specifically, each
    sub-directory of `data_dir` corresponds to a label in the dataset (such as 'cat' and 'dog' ), and named by the
    label. Such a subdirectory contains files (such as images), each of which correspond to a data record with this
    label.

    When `extractor` is `None`, each data record is stored in its original encoding, e.g., a JPEG image. The `extractor`
    can be an arbitrary Python function that transforms an original data record, e.g., from sound to image.

    If the output TFRecords file already exists, it is skipped unless `overwrite` is `True`.

    :param data_dir: Directory containing data files, whose contents are to be put in the output TFRecords files.
    :param output_file: Base output file name, such as `data.tfrec`. An ouptut file name is the base file name plus the
                        shard ID and total number of shards, such as `data.tfrec00000-of-00005`
    :param shuffle: Whether or not to shuffle the data records. Default: no shuffle.
    :param overwrite: Whether or not to overwrite, when the output file already exists.
    :param extractor: A Python function that transforms a data file. Its outputs are added to the output TFRecords.
                      If `extractor` is `None`, the original contents of the data file is added to the output TFRecords.
    :param file_ext: Extension of input data files. Default: 'jpg'.
    :param exclude_dirs: List of subdirectories to exclude, which do not contain input data files. Default: empty list.
    :return: `paths`: list of paths to all input data files. `y`: list of labels, each of which is an integer. `labels`:
             list of string labels, such as the first corresponds to `y=0`, the second corresponds to `y=1`, and so on.
    """
    labels = sub_dirs(data_dir, exclude_dirs)
    paths, y = find_files(data_dir, labels, shuffle=shuffle, file_ext=file_ext)
    files_tfrecord(paths, y, output_file, overwrite, extractor)

    return paths, y, labels


# todo: add number of cores
def data_dir_tfrecord_shards(data_dir: str, output_file: str, shuffle: bool = False, overwrite: bool = False,
                             extractor=None, file_ext: str = 'jpg', exclude_dirs: List[str] = [],
                             num_shards: int = 2) -> Tuple[List[str], List[int], List[str]]:
    """
    Create a number of TFRecords data files from the contents of a data directory `data_dir`. Specifically, each
    sub-directory of `data_dir` corresponds to a label in the dataset (such as 'cat' and 'dog' ), and named by the
    label. Such a subdirectory contains files (such as images), each of which correspond to a data record with this
    label.

    When `extractor` is `None`, each data record is stored in its original encoding, e.g., a JPEG image. The `extractor`
    can be an arbitrary Python function that transforms an original data record, e.g., from sound to image.

    Each output TFRecord file is s shard of the whole dataset. Each shard contains roughly the same number of data
    records (e.g., images).

    If an output TFRecords file already exists, it is skipped unless `overwrite` is `True`.

    :param data_dir: Directory containing data files, whose contents are to be put in the output TFRecords files.
    :param output_file: Base output file name, such as `data.tfrec`. An ouptut file name is the base file name plus the
                        shard ID and total number of shards, such as `data.tfrec00000-of-00005`
    :param shuffle: Whether or not to shuffle the data records. Should be set to True for cross validation.
                    Default: False (no shuffle).
    :param overwrite: Whether or not to overwrite, when the output file already exists.
    :param extractor: A Python function that transforms a data file. Its outputs are added to the output TFRecords.
                      If `extractor` is `None`, the original contents of the data file is added to the output TFRecords.
    :param file_ext: Extension of input data files. Default: 'jpg'.
    :param exclude_dirs: List of subdirectories to exclude, which do not contain input data files. Default: empty list.
    :param num_shards: Number of shards.
    :return: `paths`: list of paths to all input data files. `y`: list of labels, each of which is an integer. `labels`:
             list of string labels, such as the first corresponds to `y=0`, the second corresponds to `y=1`, and so on.
    """
    labels = sub_dirs(data_dir, exclude_dirs)
    paths, y = find_files(data_dir, labels, shuffle=shuffle, file_ext=file_ext)

    spacing = np.linspace(0, len(y), num_shards + 1).astype(np.int)
    ranges = [[spacing[i], spacing[i + 1]] for i in range(num_shards)]
    threads = []

    for i in range(num_shards):
        start, end = ranges[i][0], ranges[i][1]
        args = (paths[start:end], y[start:end],
                f'{output_file}-{i:05d}-of-{num_shards:05d}', overwrite, extractor)
        t = threading.Thread(target=files_tfrecord, args=args)
        t = threading.Thread(target=files_tfrecord, args=args)
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)
    return paths, y, labels


def data_dir_label_csv_tfrecord(data_dir: str, csv_fn: str, output_file: str, shuffle: bool = False,
                                overwrite: bool = False, extractor=None, file_ext: str = 'jpg') -> Tuple[
    List[str], List[int], List[str]]:
    paths, y, labels = find_files_with_label_csv(data_dir, csv_fn, shuffle=shuffle, file_ext=file_ext)
    files_tfrecord(paths, y, output_file, overwrite, extractor)

    return paths, y, labels


def tfexample_raw_parser(example) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a given TFExample containing an (image, label) pair, whose contents are undefined.
    :param example: An input TFExample.
    :return: Parsed image and label Tensors.
    """
    feat_dict = {'image': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(example, features=feat_dict)
    return feat['image'], feat['label']


def tfexample_numpy_image_parser(example, h: int, w: int, c: int = 3, dtype=tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a given TFExample containing an (image, label) pair, where the image is represented as an 3D array of shape
    [h*w*c] (i.e., flattened).

    :param example: An input TFExample.
    :param h: Height of the image.
    :param w: Weight of the image.
    :param c: Number of color channels. Default to 3 (RGB).
    :param dtype: Data type of the returned image.
    :return: Parsed image and label Tensors.
    """
    feat_dict = {'image': tf.FixedLenFeature([h * w * c], dtype),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(example, features=feat_dict)
    x, y = feat['image'], feat['label']
    x = tf.reshape(x, [h, w, c])
    return x, y


# todo: add dtype as a parameter.
def tfexample_image_parser(example, tfms: List) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a given TFExample containing an encoded image (such as JPEG) and a label. Then apply the given sequence of
    transformations.

    :param example: An input TFExample.
    :param tfms: A sequence of transforms.
    :return: Parsed image and label Tensors.
    """
    x, y = tfexample_raw_parser(example)
    x = tf.image.decode_image(x, channels=3, dtype=tf.float32)
    x = apply_transforms(x, tfms)
    return x, y


def get_tfexample_image_parser(h: int, w: int, training: bool = True, normalizer=imagenet_normalize_tf):
    """
    Get a image parser function that parses a TFRecord into an image, applying standard transformations for ImageNet.
    For the training set, standard ImageNet data augmentation is also applied.

    :param h: Height of a data image.
    :param w: Width of a data image.
    :param training: Whether this is a training set.
    :param normalizer: Data normalization function. Default to Tensorflow's ImageNet noramlization function, i.e.,
    `x = (x-0.5)*2`.
    :return: A function that parses a TFExample into an image.
    """
    tfms = get_train_transforms(h, w, normalizer) if training else get_eval_transforms(h, w, normalizer=normalizer)
    return lambda example: tfexample_image_parser(example, tfms)


def tfrecord_fetch_dataset(fn: str):
    """
    Create a `tf.data` dataset from a given TFRecord file name.

    :param fn: Name of the TFRecord file.
    :return: `tf.data` dataset.
    """
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(fn, buffer_size=buffer_size)
    return dataset


def crossval_ds(dataset, n_folds: int, val_fold_idx: int, training: bool = True):
    """
    Partition a given `tf.data` dataset into training and validation sets, according to k-fold cross validation
    requirements.

    :param dataset: A given `tf.data` dataset containing the whole dataset.
    :param n_folds: Number of cross validation folds.
    :param val_fold_idx: Fold ID for validation set, in cross validation.
    :param training: Whether to return training or validation data.
    :return: either training or validation dataset.
    """
    if training:
        trn_shards = itertools.chain(range(val_fold_idx), range(val_fold_idx + 1, n_folds))
        update_func = lambda ds, i: ds.concatenate(dataset.shard(n_folds, i))
        dataset = functools.reduce(update_func, trn_shards, dataset.shard(n_folds, next(trn_shards)))
    else:
        dataset = dataset.shard(n_folds, val_fold_idx)
    return dataset


def tfrecord_ds(file_pattern: str, parser, batch_size: int, training: bool = True, shuffle_buf_sz: int = 50000,
                n_cores: int = 2, n_folds: int = 1, val_fold_idx: int = 0):
    """
    Create a `tf.data` input pipeline from TFRecords files whose names satisfying a given pattern. Optionally partitions
    the data into training and validation sets according to k-fold cross-validation requirements.

    :param file_pattern: file pattern such as `data_train*.tfrec`
    :param parser: TFRecords parser function, which may also perform data augmentations.
    :param batch_size: Size of a data batch.
    :param training: Whether this is a training dataset, in which case the dataset is randomly shuffled and repeated.
    :param shuffle_buf_sz: Shuffle buffer size, for shuffling a training dataset. Default: 50k records.
    :param n_cores: Number of CPU cores, i.e., parallel threads.
    :param n_folds: Number of cross validation folds. Default: 1, meaning no cross validation.
    :param val_fold_idx: Fold ID for validation set, in cross validation. Ignored when `n_folds` is 1.
    :return: a `tf.data` dataset satisfying the above descriptions.
    """
    if not file_pattern.startswith('gs://'):
        # under construction
        dataset = tpu_datasets.StreamingFilesDataset(file_pattern, filetype='tfrecord',
                                                     batch_transfer_size=batch_size)
    else:
        dataset = tf.data.Dataset.list_files(file_pattern)
        fetcher = tf.data.experimental.parallel_interleave(tfrecord_fetch_dataset, cycle_length=n_cores, sloppy=True)
        dataset = dataset.apply(fetcher)

    mapper_batcher = tf.data.experimental.map_and_batch(parser, batch_size=batch_size, num_parallel_batches=n_cores,
                                                        drop_remainder=True)

    if n_folds > 1:
        dataset = crossval_ds(dataset, n_folds, val_fold_idx, training)

    if training:
        dataset = dataset.shuffle(shuffle_buf_sz)
        dataset = dataset.repeat()

    dataset = dataset.apply(mapper_batcher)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def numpy_ds(x, y, batch_size: int, training: bool = True, shuffle_buf_sz: int = 50000, n_folds: int = 1,
             val_fold_idx: int = 0):
    """
    Create a `tf.data` input pipeline from numpy arrays `x` and `y`. Optionally partitions the data into training and
    validation sets according to k-fold cross validation requirements.

    :param x: Data attributes, such as images.
    :param y: Data labels.
    :param batch_size: Size of a data batch.
    :param training: Whether this is a training dataset, in which case the dataset is randomly shuffled and repeated.
    :param shuffle_buf_sz: Shuffle buffer size, for shuffling a training dataset. Default: 50k records.
    :param n_folds: Number of cross validation folds. Default: 1, meaning no cross validation.
    :param val_fold_idx: Fold ID for validation set, in cross validation. Ignored when `n_folds` is 1.
    :return: a `tf.data` dataset satisfying the above descriptions.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if n_folds > 1:
        dataset = crossval_ds(dataset, n_folds, val_fold_idx, training)

    if training:
        dataset = dataset.shuffle(shuffle_buf_sz)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def get_model_dir(bucket: str, model: str) -> str:
    """
    Get recommended directory to store parameters of a pre-trained model.

    :param bucket: Google Cloud Storage bucket.
    :param model: Name of the pre-trained model.
    :return: GCS path to store the pre-trained model.
    """
    return os.path.join(os.path.join(bucket, 'model'), model)


def get_gcs_dirs(bucket: str, project: str) -> Tuple[str, str]:
    """
    Get recommended directories for storing datasets (data_dir) and intermediate files generated during training
    (work_dir).

    :param bucket: Google Cloud Storage bucket.
    :param project: Name of the project.
    :return: Data directory for storaing datasets, and work directory for storing intermediate files.
    """
    data_dir = os.path.join(os.path.join(bucket, 'data'), project)
    work_dir = os.path.join(os.path.join(bucket, 'work'), project)
    return data_dir, work_dir
