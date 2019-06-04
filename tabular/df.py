from ..imports import *


def b_gb(b: int) -> float:
    return round(b / 1e9, 2)


def reduce_mem(df: pd.DataFrame, print_info: bool = False) -> pd.DataFrame:
    original_memory = None
    if print_info:
        original_memory = df.memory_usage().sum()

    for col in df:
        col_type = df[col].dtype
        if col_type == 'object' and df[col].nunique() < df.shape[0]:
            df[col] = df[col].astype('category')
        elif list(df[col].unique()) == [1, 0] or list(df[col].unique()) == [0, 1]:
            df[col] = df[col].astype(bool)
        elif str(col_type)[:3] == 'int':
            cmin = df[col].min()
            cmax = df[col].max()
            # Can use unsigned int here too
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        elif str(col_type)[:5] == 'float':
            cmin = df[col].min()
            cmax = df[col].max()
            if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    if print_info:
        new_memory = df.memory_usage().sum()
        tf.logging.INFO(f'Original Memory Usage: {b_gb(original_memory)} GB.')
        tf.logging.INFO(f'New Memory Usage: {b_gb(new_memory)} GB.')

    return df


def fillna_cat(df: pd.DataFrame, default_value=-9999):
    for c in df.select_dtypes('category'):
        s = df[c]
        s.cat.add_categories([default_value], inplace=True)
    df.fillna(default_value, inplace=True)


def flatten_columns(df: pd.DataFrame, key_col: str, prefix: str):
    columns = [key_col]
    for var in df.columns.levels[0]:
        if var != key_col:
            for stat in df.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (prefix, var, stat))
    df.columns = columns


def csv_to_pickle(data_dir: str, df_name: str):
    df = pd.read_csv(os.path.join(data_dir, df_name + '.csv'))
    df = reduce_mem(df, True)
    df.to_pickle(os.path.join(data_dir, df_name, '.pkl'))
