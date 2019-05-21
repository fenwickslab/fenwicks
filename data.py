from .imports import *

import threading
import itertools

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

from . import io

from tensorflow.contrib.tpu.python.tpu import datasets as tpu_datasets


def float_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_tffeature(value) -> tf.train.Feature:
    if isinstance(value, str):
        value = bytes(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def raw_image_tfexample(raw_image, y=None) -> tf.train.Example:
    if y is None:
        feat_dict = {'image': bytes_tffeature(raw_image)}
    else:
        feat_dict = {'image': bytes_tffeature(raw_image), 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def numpy_tfexample(x, y=None) -> tf.train.Example:
    if y is None:
        feat_dict = {'image': float_tffeature(x.tolist())}
    else:
        feat_dict = {'image': float_tffeature(x.tolist()), 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))


def numpy_tfrecord(output_fn: str, X, y=None, overwrite: bool = False):
    n = X.shape[0]
    X_reshape = X.reshape(n, -1)

    if overwrite or not tf.io.gfile.exists(output_fn):
        with tf.io.TFRecordWriter(output_fn) as record_writer:
            for i in tqdm_notebook(range(n)):
                example = numpy_tfexample(X_reshape[i]) if y is None else numpy_tfexample(X_reshape[i], y[i])
                record_writer.write(example.SerializeToString())
    else:
        tf.logging.info('Output file already exists. Skipping.')


# todo: number of threads.
def numpy_tfrecord_shards(output_fn: str, x, y=None, num_shards: int = 2):
    spacing = np.linspace(0, len(x), num_shards + 1).astype(np.int)
    ranges = [[spacing[i], spacing[i + 1]] for i in range(num_shards)]

    threads = []
    for i in range(num_shards):
        start, end = ranges[i][0], ranges[i][1]
        args = (x[start:end], y[start:end],
                f'{output_fn}-{i:05d}-of-{num_shards:05d}')
        t = threading.Thread(target=numpy_tfrecord, args=args)
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)


def files_tfrecord(output_fn: str, paths: List[str], y: List[int] = None, overwrite: bool = False,
                   extractor: Callable = None):
    """
    Create a TFRecord file that contains the contents of a given list of files, and optionally their corresponding
    labels. The contents of the given files can be transformed through an extractor function.

    :param output_fn: File name of the TFRecord file.
    :param paths: List of input files.
    :param y: (Option) Labels, one for each input file. Default: no label provided.
    :param overwrite: Whether to overwrite the output TFRecord file when it already exists. Default: False.
    :param extractor: A function that takes a file as input and outputs a numpy array containing the transformed file
                      content. Default: no extractor, in which case the raw file contents are written to the output
                      TFRecord file.
    :return: None.
    """
    if overwrite or not tf.io.gfile.exists(output_fn):
        with tf.io.TFRecordWriter(output_fn) as record_writer:
            for i, path in enumerate(tqdm_notebook(paths)):
                if extractor is None:
                    with tf.io.gfile.GFile(path, 'rb') as f:
                        img = f.read()
                        example = raw_image_tfexample(img) if y is None else raw_image_tfexample(img, y[i])
                else:
                    img = extractor(path)
                    img = img.reshape(-1)
                    example = numpy_tfexample(img) if y is None else numpy_tfexample(img, y[i])
                record_writer.write(example.SerializeToString())
    else:
        tf.logging.info('Output file already exists. Skipping.')


def data_dir_tfrecord(data_dir: str, output_fn: str, shuffle: bool = False, overwrite: bool = False,
                      extractor: Callable = None, file_ext: str = 'jpg', exclude_dirs: List[str] = None) -> Tuple[
    List[str], List[int], List[str]]:
    """
    Create a TFRecords data file from the contents of a data directory `data_dir`. Specifically, each
    sub-directory of `data_dir` corresponds to a label in the dataset (such as 'cat' and 'dog' ), and named by the
    label. Such a subdirectory contains files (such as images), each of which correspond to a data record with this
    label.

    When `extractor` is `None`, each data record is stored in its original encoding, e.g., a JPEG image. The `extractor`
    can be an arbitrary Python function that transforms an original data record, e.g., from sound to image.

    If the output TFRecords file already exists, it is skipped unless `overwrite` is `True`.

    :param data_dir: Directory containing data files, whose contents are to be put in the output TFRecords files.
    :param output_fn: Base output file name, such as `data.tfrec`. An ouptut file name is the base file name plus the
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
    exclude_dirs = exclude_dirs or []
    labels = io.sub_dirs(data_dir, exclude_dirs)
    paths, y = io.find_files(data_dir, labels, shuffle=shuffle, file_ext=file_ext)
    files_tfrecord(output_fn, paths, y, overwrite, extractor)

    return paths, y, labels


def to_tfrecord_split(paths: List[str], y: List[int], labels: List[str], train_fn: str, valid_fn: str,
                      valid_pct: float = 0.2, split_rand_state=777, overwrite: bool = False,
                      extractor: Callable = None) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    n_valid = int(len(y) * valid_pct) // 8 * 8
    paths_train, paths_valid, y_train, y_valid = train_test_split(paths, y, test_size=n_valid,
                                                                  random_state=split_rand_state)
    files_tfrecord(train_fn, paths_train, y_train, overwrite, extractor)
    files_tfrecord(valid_fn, paths_valid, y_valid, overwrite, extractor)
    return paths_train, paths_valid, y_train, y_valid, labels


def data_dir_tfrecord_split(data_dir: str, train_fn: str, valid_fn: str, valid_pct: float = 0.2, split_rand_state=777,
                            overwrite: bool = False, extractor: Callable = None, file_ext: str = 'jpg',
                            exclude_dirs: List[str] = None) -> Tuple[
    List[str], List[int], List[str], List[int], List[str]]:
    exclude_dirs = exclude_dirs or []
    labels = io.sub_dirs(data_dir, exclude_dirs)
    paths, y = io.find_files(data_dir, labels, file_ext=file_ext)
    return to_tfrecord_split(paths, y, labels, train_fn, valid_fn, valid_pct, split_rand_state, overwrite, extractor)


# todo: add number of cores
def data_dir_tfrecord_shards(data_dir: str, output_fn: str, shuffle: bool = False, overwrite: bool = False,
                             extractor: Callable = None, file_ext: str = 'jpg', exclude_dirs: List[str] = None,
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
    :param output_fn: Base output file name, such as `data.tfrec`. An ouptut file name is the base file name plus the
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
    exclude_dirs = exclude_dirs or []

    labels = io.sub_dirs(data_dir, exclude_dirs)
    paths, y = io.find_files(data_dir, labels, shuffle=shuffle, file_ext=file_ext)

    spacing = np.linspace(0, len(y), num_shards + 1).astype(np.int)
    ranges = [[spacing[i], spacing[i + 1]] for i in range(num_shards)]
    threads = []

    for i in range(num_shards):
        start, end = ranges[i][0], ranges[i][1]
        args = (paths[start:end], y[start:end],
                f'{output_fn}-{i:05d}-of-{num_shards:05d}', overwrite, extractor)
        t = threading.Thread(target=files_tfrecord, args=args)
        t.start()
        threads.append(t)

    tf.train.Coordinator().join(threads)
    return paths, y, labels


def data_dir_label_csv_tfrecord(data_dir: str, csv_fn: str, output_fn: str, shuffle: bool = False,
                                overwrite: bool = False, extractor: Callable = None, file_ext: str = 'jpg', id_col='id',
                                label_col='label', _labels: List[str] = None) -> Tuple[List[str], List[int], List[str]]:
    paths, y, labels = io.find_files_with_label_csv(data_dir, csv_fn, shuffle=shuffle, file_ext=file_ext, id_col=id_col,
                                                    label_col=label_col, _labels=_labels)
    files_tfrecord(output_fn, paths, y, overwrite, extractor)

    return paths, y, labels


def data_dir_re_tfrecord(data_dir: str, pat: str, output_fn: str, shuffle: bool = False, overwrite: bool = False,
                         extractor: Callable = None, file_ext: str = 'jpg') -> Tuple[List[str], List[int], List[str]]:
    paths = io.find_files_no_label(data_dir, shuffle, file_ext)
    labels, y = io.extract_labels_re(pat, paths)
    files_tfrecord(output_fn, paths, y, overwrite, extractor)
    return paths, y, labels


def data_dir_re_tfrecord_split(data_dir: str, pat: str, train_fn: str, valid_fn: str, valid_pct: float = 0.2,
                               split_rand_state=777, overwrite: bool = False, extractor: Callable = None,
                               file_ext: str = 'jpg') -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    paths = io.find_files_no_label(data_dir, file_ext=file_ext)
    labels, y = io.extract_labels_re(pat, paths)
    return to_tfrecord_split(paths, y, labels, train_fn, valid_fn, valid_pct, split_rand_state, overwrite, extractor)


def data_dir_no_label_tfrecord(data_dir: str, output_fn: str, shuffle: bool = False,
                               overwrite: bool = False, extractor: Callable = None, file_ext: str = 'jpg') -> List[str]:
    """
    Create a TFRecords data file from the contents of a data directory `data_dir`, which contain data files with a given
    file extension specified in `file_ext`. No labels are given for these data files, i.e., this is an unlabeled test
    dataset.

    When `extractor` is `None`, each data record is stored in its original encoding, e.g., a JPEG image. The `extractor`
    can be an arbitrary Python function that transforms an original data record, e.g., from sound to image.

    If the output TFRecords file already exists, it is skipped unless `overwrite` is `True`.

    :param data_dir: Directory containing data files, whose contents are to be put in the output TFRecords files.
    :param output_fn: Base output file name, such as `data.tfrec`. An ouptut file name is the base file name plus the
                        shard ID and total number of shards, such as `data.tfrec00000-of-00005`
    :param shuffle: Whether or not to shuffle the data records. Default: no shuffle.
    :param overwrite: Whether or not to overwrite, when the output file already exists.
    :param extractor: A Python function that transforms a data file. Its outputs are added to the output TFRecords.
                      If `extractor` is `None`, the original contents of the data file is added to the output TFRecords.
    :param file_ext: Extension of input data files. Default: 'jpg'.
    :return: list of paths to all input data files.

    """
    paths = io.find_files_no_label(data_dir, shuffle, file_ext)
    files_tfrecord(output_fn, paths, overwrite=overwrite, extractor=extractor)
    return paths


def tfexample_raw_parser(tfexample: tf.train.Example, has_label: bool = True):
    """
    Parse a given TFExample containing an (image, label) pair, whose contents are undefined.
    :param tfexample: An input TFExample.
    :param has_label: Whether the input TFExample contains a label. Default: True.
    :return: Parsed image (and optionally label) Tensors.
    """

    if has_label:
        feat_dict = {'image': tf.FixedLenFeature([], tf.string),
                     'label': tf.FixedLenFeature([], tf.int64)}
    else:
        feat_dict = {'image': tf.FixedLenFeature([], tf.string)}

    feat = tf.parse_single_example(tfexample, features=feat_dict)

    if has_label:
        return feat['image'], feat['label']
    else:
        return feat['image']


def tfexample_numpy_image_parser(tfexample: tf.train.Example, h: int, w: int, c: int = 3, dtype=tf.float32) -> Tuple[
    tf.Tensor, tf.Tensor]:
    """
    Parse a given TFExample containing an (image, label) pair, where the image is represented as an 3D array of shape
    [h*w*c] (i.e., flattened).
    :param tfexample: An input TFExample.
    :param h: Height of the image.
    :param w: Weight of the image.
    :param c: Number of color channels. Default to 3 (RGB).
    :param dtype: Data type of the returned image.
    :return: Parsed image and label Tensors.
    """
    feat_dict = {'image': tf.FixedLenFeature([h * w * c], dtype),
                 'label': tf.FixedLenFeature([], tf.int64)}
    feat = tf.parse_single_example(tfexample, features=feat_dict)
    x, y = feat['image'], feat['label']
    x = tf.reshape(x, [h, w, c])
    return x, y


# todo: add dtype as a parameter.
def tfexample_image_parser(tfexample: tf.train.Example, tfms: List[Callable] = None, has_label: bool = True):
    """
    Parse a given TFExample containing an encoded image (such as JPEG) and a label. Then apply the given sequence of
    transformations.

    :param tfexample: An input TFExample.
    :param tfms: A sequence of transforms.
    :param has_label: Whether the input TFExample contains a label. Default: True.
    :return: Parsed image (and optionally also label) Tensors.
    """

    parsed_example = tfexample_raw_parser(tfexample, has_label)

    if has_label:
        x, y = parsed_example
    else:
        x = parsed_example
        y = None

    x = tf.image.decode_image(x, channels=3, dtype=tf.float32)
    if tfms is not None:
        x = core.sequential_transforms(x, tfms)

    if has_label:
        return x, y
    else:
        return x


def get_tfexample_image_parser(tfms: List[Callable], has_label: bool = True) -> Callable:
    return functools.partial(tfexample_image_parser, tfms=tfms, has_label=has_label)


def tfrecord_fetch_dataset(fn: str) -> tf.data.Dataset:
    """
    Create a `tf.data` dataset from a given TFRecord file name.

    :param fn: Name of the TFRecord file.
    :return: `tf.data` dataset.
    """
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(fn, buffer_size=buffer_size)
    return dataset


def crossval_ds(dataset, n_folds: int, val_fold_idx: int, training: bool = True) -> tf.data.Dataset:
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
                n_cores: int = 2, n_folds: int = 1, val_fold_idx: int = 0, streaming: bool = False) -> tf.data.Dataset:
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
    :param streaming: under construction.
    :return: a `tf.data` dataset satisfying the above descriptions.
    """
    if streaming:
        # fixme
        dataset = tpu_datasets.StreamingFilesDataset(file_pattern, filetype='tfrecord', batch_transfer_size=batch_size)
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


def numpy_ds(xs: Union[np.ndarray, List], ys: Union[np.ndarray, List], batch_size: int, training: bool = True,
             shuffle_buf_sz: int = 50000, n_folds: int = 1, val_fold_idx: int = 0) -> tf.data.Dataset:
    """
    Create a `tf.data` input pipeline from numpy arrays `x` and `y`. Optionally partitions the data into training and
    validation sets according to k-fold cross validation requirements.

    :param xs: Data attributes, such as images.
    :param ys: Data labels.
    :param batch_size: Size of a data batch.
    :param training: Whether this is a training dataset, in which case the dataset is randomly shuffled and repeated.
    :param shuffle_buf_sz: Shuffle buffer size, for shuffling a training dataset. Default: 50k records.
    :param n_folds: Number of cross validation folds. Default: 1, meaning no cross validation.
    :param val_fold_idx: Fold ID for validation set, in cross validation. Ignored when `n_folds` is 1.
    :return: a `tf.data` dataset satisfying the above descriptions.
    """
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))

    if n_folds > 1:
        dataset = crossval_ds(dataset, n_folds, val_fold_idx, training)

    if training:
        dataset = dataset.shuffle(shuffle_buf_sz)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
