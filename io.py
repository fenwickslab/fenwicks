from .vision.transform import *
import numpy as np
import random
import threading
import os

from typing import List, Tuple
from tqdm import tqdm_notebook as tqdm


def enum_files(data_dir: str, file_ext: str = 'jpg') -> List[str]:
    file_pattern = os.path.join(data_dir, f'*.{file_ext}')
    matching_files = tf.gfile.Glob(file_pattern)
    return matching_files


def find_files(data_dir: str, labels: List[str], shuffle=False, file_ext: str = 'jpg') -> Tuple[List[str], List[int]]:
    filepaths = []
    filelabels = []

    for i, label in enumerate(labels):
        matching_files = enum_files(os.path.join(data_dir, label), file_ext)
        filepaths.extend(matching_files)
        filelabels.extend([i] * len(matching_files))

    if shuffle:
        c = list(zip(filepaths, filelabels))
        random.shuffle(c)
        filepaths, filelabels = zip(*c)
        filepaths, filelabels = list(filepaths), list(filelabels)

    return filepaths, filelabels


def create_clean_dir(path: str):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MkDir(path)


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


def numpy_tfexample(X, y):
    feat_dict = {'image': float_tffeature(X.tolist()),
                 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def raw_image_tfexample(raw_image, y):
    feat_dict = {'image': bytes_tffeature(raw_image),
                 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def numpy_tfrecord(X, y, output_file: str):
    n = X.shape[0]
    X_reshape = X.reshape(n, -1)

    with tf.io.TFRecordWriter(output_file) as record_writer:
        for i in tqdm(range(n)):
            example = numpy_tfexample(X_reshape[i], y[i])
            record_writer.write(example.SerializeToString())


def numpy_tfrecord_shards(X, y, output_file: str, num_shards: int = 2):
    spacing = np.linspace(0, len(X), num_shards + 1).astype(np.int)
    ranges = [[spacing[i], spacing[i + 1]] for i in range(num_shards)]

    threads = []
    for i in range(num_shards):
        start, end = ranges[i][0], ranges[i][1]
        args = (X[start:end], y[start:end],
                f'{output_file}-{i:05d}-of-{num_shards:05d}')
        t = threading.Thread(target=numpy_tfrecord, args=args)
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)


def files_tfrecord(paths: List[str], y: List[int], output_file: str, overwrite=False, extractor=None):
    if not overwrite and not tf.gfile.Exists(output_file):
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
                      file_ext: str = 'jpg', exclude_dirs: List[str] = []):
    labels = sub_dirs(data_dir, exclude_dirs)
    paths, y = find_files(data_dir, labels, shuffle=shuffle, file_ext=file_ext)
    files_tfrecord(paths, y, output_file, overwrite, extractor)

    return paths, y, labels


def data_dir_tfrecord_shards(data_dir: str, output_file: str, shuffle: bool = False, overwrite: bool = False,
                             extractor=None, file_ext: str = 'jpg', exclude_dirs: List[str] = [], num_shards: int = 2):
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
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)
    return paths, y, labels


def tfexample_numpy(example, h: int, w: int, c: int = 3):
    d = h * w * c
    feat_dict = {'image': tf.FixedLenFeature([d], tf.float32),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(example, features=feat_dict)
    x, y = feat['image'], feat['label']
    x = tf.reshape(x, [h, w, c])
    return x, y


def tfexample_image_parser(example, tfms):
    feat_dict = {'image': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(example, features=feat_dict)
    x, y = feat['image'], feat['label']

    x = tf.image.decode_image(x, channels=3, dtype=tf.float32)
    x = apply_transforms(x, tfms)
    return x, y


def get_tfexample_image_parser(h: int, w: int, training: bool = True, normalizer=imagenet_normalize_tf):
    tfms = get_train_transforms(h, w, normalizer) if training else get_eval_transforms(h, w, normalizer=normalizer)
    return lambda example: tfexample_image_parser(example, tfms)


def tfrecord_fetch_dataset(fn: str):
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(fn, buffer_size=buffer_size)
    return dataset


def tfrecord_ds(file_pattern: str, parser, batch_size: int, training=True, shuffle_buf_sz: int = 50000,
                n_cores: int = 2, n_folds: int = 1, val_fold_idx: int = 0):
    dataset = tf.data.Dataset.list_files(file_pattern)
    fetcher = tf.data.experimental.parallel_interleave(tfrecord_fetch_dataset, cycle_length=n_cores, sloppy=True)
    mapper_batcher = tf.data.experimental.map_and_batch(parser, batch_size=batch_size, num_parallel_batches=n_cores,
                                                        drop_remainder=True)
    dataset = dataset.apply(fetcher)

    if n_folds > 1:
        if training:
            trn = None
            for i in range(0, n_folds):
                if i != val_fold_idx:
                    trn = dataset.shard(n_folds, i) if trn is None else trn.concatenate(dataset.shard(n_folds, i))
            dataset = trn
        else:
            dataset = dataset.shard(n_folds, val_fold_idx)

    if training:
        dataset = dataset.shuffle(shuffle_buf_sz)
        dataset = dataset.repeat()

    dataset = dataset.apply(mapper_batcher)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def tfrecord_numpy(file_pattern: str, n: int, h: int, w: int, c: int = 3):
    parser = lambda x: tfexample_numpy(x, h, w, c)
    ds = tfrecord_ds(file_pattern, parser, n)
    return ds.make_one_shot_iterator().next()


def get_gcs_dirs(bucket: str, project: str, model: str = 'custom') -> Tuple[str, str, str]:
    data_dir = os.path.join(os.path.join(bucket, 'data'), project)
    work_dir = os.path.join(os.path.join(bucket, 'work'), project)
    model_dir = os.path.join(os.path.join(bucket, 'model'), model)
    return data_dir, work_dir, model_dir
