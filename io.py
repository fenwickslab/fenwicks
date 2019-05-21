from .imports import *

import urllib.request

from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder


def enum_files(data_dir: str, file_ext: str = 'jpg') -> List[str]:
    """
    Enumerate all files with a given extension in a given data directory.

    :param data_dir: Data directory.
    :param file_ext: Extensions of files to enumerate. Default: 'jpg'.
    :return: A list of file names. Note that these are base file names, not full paths.
    """
    file_pattern: str = os.path.join(data_dir, f'*.{file_ext}')
    matching_files: List[str] = gfile.glob(file_pattern)
    return matching_files


def find_files(data_dir: str, labels: List[str], shuffle: bool = False, file_ext: str = 'jpg') -> Tuple[
    List[str], List[int]]:
    """
    Find all input files wth a given extension in specific subdirectories of a given data directory. Optionally shuffle
    the found file names.

    :param data_dir: Data directory.
    :param labels: Set of subdirectories (named after labels) in which to find files.
    :param shuffle: Whether to shuffle the outputs.
    :param file_ext: File extension. ONly find files with this extension.
    :return: Two lists: one for file paths and the other for their corresponding labels represented as indexes.
    """
    filepaths: List[str] = []
    filelabels: List[int] = []

    for i, label in enumerate(labels):
        matching_files = enum_files(os.path.join(data_dir, label), file_ext)
        filepaths.extend(matching_files)
        filelabels.extend([i] * len(matching_files))

    if shuffle:
        filepaths, filelabels = core.shuffle_lists(filepaths, filelabels)

    return filepaths, filelabels


def find_files_with_label_csv(data_dir: str, csv_fn: str, shuffle: bool = False, file_ext: str = 'jpg', id_col='id',
                              label_col='label', _labels: List[str] = None) -> Tuple[List[str], List[int], List[str]]:
    train_labels: pd.DataFrame = pd.read_csv(csv_fn)
    labels = _labels or sorted(train_labels[label_col].unique())
    key_id = dict([(label, idx) for idx, label in enumerate(labels)])

    filepaths = []
    filelabels = []

    for _, row in train_labels.iterrows():
        filepaths.append(os.path.join(data_dir, f'{row[id_col]}.{file_ext}'))
        filelabels.append(key_id[row[label_col]])

    if shuffle:
        filepaths, filelabels = core.shuffle_lists(filepaths, filelabels)

    return filepaths, filelabels, labels


def find_files_no_label(data_dir: str, shuffle: bool = False, file_ext: str = 'jpg') -> List[str]:
    """
    Get all files with a given extension in a data directory.

    :param data_dir: Data directory.
    :param shuffle: Whether to shuffle the resulting file paths.
    :param file_ext: File extension.
    :return: List of file paths.
    """
    filepaths: List[str] = enum_files(data_dir, file_ext)
    if shuffle:
        random.shuffle(filepaths)
    return filepaths


def extract_labels_re(pat: str, filepaths: List[str]) -> Tuple[List[str], List[int]]:
    """
    Extract labels and class names from a list of file paths, using a regular expression.

    :param pat: Regular expression to extract the label from a file path. The first matching group is the label.
    :param filepaths: List of file paths.
    :return: (i) list of class names, (ii) list of integer labels, in the same order of the file paths.
    """
    pat = re.compile(pat)
    labels = list(map(lambda x: pat.search(x).group(1), filepaths))
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return le.classes_, labels


def create_clean_dir(path: str):
    """
    Create a new directory specified by `path`. If this directory already exists, delete all its files and
    subdirectories.

    :param path: Path to the directory to be created or cleaned.
    :return: None
    """
    if gfile.exists(path):
        gfile.rmtree(path)
    gfile.makedirs(path)


def file_size(fn: str) -> int:
    """
    Get the size of a file in bytes. Works for files on Google Cloud Storage.
    :param fn: Path to the file.
    :return: Size of the file.
    """
    stat = gfile.stat(fn)
    return stat.length


def download(url: str, fn: str, overwrite: bool = False):
    if overwrite or not gfile.exists(fn):
        urllib.request.urlretrieve(url, fn)
    else:
        tf.logging.info(f'Destination file exists. Skipping.')


def unzip(fn, dest_dir: str = '.', overwrite: bool = False):
    """
    Extract one or more .zip or .7z file(s) to a destination directory.

    :param fn: Name of the file(s) to be decompressed. The type of `fn` can be either `str`, or `List[str]`
    :param dest_dir: Destination directory. Default: current directory.
    :param overwrite: Whether to overwrite when the destination directory already exists. Default: False, in which case
                      nothing is done when the destination directory already exists.
    :return: None.
    """

    try:
        import libarchive.public
    except ImportError:
        raise ImportError('libarchive not installed. Run !apt install libarchive-dev and then !pip install libarchive.')

    is_one_file: bool = isinstance(fn, str)

    if overwrite or not gfile.exists(dest_dir):
        gfile.makedirs(dest_dir)

        if is_one_file:
            files: List[str] = [os.path.abspath(fn)]
        else:
            files: List[str] = list(map(os.path.abspath, fn))

        cur_dir: str = os.getcwd()
        os.chdir(dest_dir)
        for fn in files:
            tf.logging.info(f'Decompressing: {fn}')
            for _ in tqdm_notebook(libarchive.public.file_pour(fn)):
                pass
        os.chdir(cur_dir)
    else:
        tf.logging.info(f'Destination directory exists. Skipping.')


def sub_dirs(data_dir: str, exclude_dirs: List[str] = None) -> List[str]:
    """
    List sub directories of a directory, except those excluded. Works for Google Cloud Storage directories.

    :param data_dir: Given directory.
    :param exclude_dirs: names (not full paths) of subdirectories to exclude.
    :return: List of subdirectories' names (not full paths).
    """

    exclude_dirs = exclude_dirs or []
    return [path for path in gfile.listdir(data_dir)
            if gfile.isdir(os.path.join(data_dir, path)) and path not in exclude_dirs]


def merge_dirs(source_dirs: List[str], dest_dir: str):
    if not gfile.exists(dest_dir):
        gfile.makedirs(dest_dir)
        for d in source_dirs:
            files = gfile.listdir(d)
            for fn in files:
                old_fn = os.path.join(d, fn)
                new_fn = os.path.join(dest_dir, fn)
                gfile.rename(old_fn, new_fn)


def get_model_dir(bucket: str, model: str) -> str:
    """
    Get recommended directory to store parameters of a pre-trained model.

    :param bucket: Google Cloud Storage bucket.
    :param model: Name of the pre-trained model.
    :return: GCS path to store the pre-trained model.
    """
    return os.path.join(bucket, 'model', model)


def get_project_dirs(root_dir: str, project: str) -> Tuple[str, str]:
    """
    Get recommended directories for storing datasets (data_dir) and intermediate files generated during training
    (work_dir).

    :param root_dir: Root directory, which is often the Google Cloud Storage bucket when using TPUs.
    :param project: Name of the project.
    :return: Data directory for storaing datasets, and work directory for storing intermediate files.
    """
    data_dir: str = os.path.join(root_dir, 'data', project)
    work_dir: str = os.path.join(root_dir, 'work', project)
    gfile.makedirs(data_dir)
    gfile.makedirs(work_dir)
    return data_dir, work_dir


# todo: gcs_path is a dir
def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Upload a local file to Google Cloud Storage, if it doesn't already exist on GCS.

    :param local_path: path to the local file to be uploaded.
    :param gcs_path: path to the GCS file. Need to be the full file name.
    :return: None.
    """
    if not gfile.exists(gcs_path):
        gfile.copy(local_path, gcs_path)
    else:
        tf.logging.info('Output file already exists. Skipping.')
