from ..imports import *

import multiprocessing as mp

from scipy.stats import kurtosis, iqr, skew
from sklearn.linear_model import LinearRegression


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
        logging.INFO(f'Original Memory Usage: {b_gb(original_memory)} GB.')
        logging.INFO(f'New Memory Usage: {b_gb(new_memory)} GB.')

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


def do_agg(feat, agg: str):
    if agg == 'sum':
        return feat.sum()
    elif agg == 'mean':
        return feat.mean()
    elif agg == 'max':
        return feat.max()
    elif agg == 'min':
        return feat.min()
    elif agg == 'std':
        return feat.std()
    elif agg == 'count':
        return feat.count()
    elif agg == 'median':
        return feat.median()
    elif agg == 'skew':
        return skew(feat)
    elif agg == 'kurt':
        return kurtosis(feat)
    elif agg == 'iqr':
        return iqr(feat)
    return None


def agg_and_merge(df: pd.DataFrame, group_cols: List[str], counted: str, agg_name: str, agg: str) -> pd.DataFrame:
    gp = df[group_cols + [counted]].groupby(group_cols)[counted]
    gp = do_agg(gp, agg)
    gp = gp.reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def get_cat_cols(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].dtype in ['object', 'category']]


def label_encoder(df: pd.DataFrame, cat_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    cat_cols = cat_cols or get_cat_cols(df)
    for col in cat_cols:
        df[col], uniques = pd.factorize(df[col])
    return df, cat_cols


def one_hot_encoder(df: pd.DataFrame, cat_cols: List[str] = None, nan_as_cat: bool = True) -> Tuple[
    pd.DataFrame, List[str]]:
    original_columns = list(df.columns)
    cat_cols = cat_cols or get_cat_cols(df)
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_cat)
    cat_cols = [c for c in df.columns if c not in original_columns]
    return df, cat_cols


def group_and_merge(df_to_agg: pd.DataFrame, df_to_merge: pd.DataFrame, prefix: str, aggregations: Dict[str, List[str]],
                    aggregate_by: str):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index([f'{prefix}{e[0]}_{e[1].upper()}' for e in agg_df.columns.tolist()])
    return df_to_merge.merge(agg_df.reset_index(), how='left', on=aggregate_by)


def add_trend_feature(df, gr, feature_name: str, prefix: str) -> pd.DataFrame:
    y = gr[feature_name].values
    x = np.arange(0, len(y)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, y)
    df[prefix + feature_name] = lr.coef_[0]
    return df


def add_agg_feats(df: pd.DataFrame, gr, feat_name: str, aggs: List[str], prefix) -> pd.DataFrame:
    for agg in aggs:
        df[f'{prefix}{feat_name}_{agg}'] = do_agg(gr[feat_name], agg)
    return df


def chunk_groups(groupby_object, chunk_size: int) -> Generator:
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func: Callable, index_name: str = 'Index', num_workers: int = 1,
                   chunk_size: int = 100000) -> pd.DataFrame:
    indices, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indices.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indices
    features.index.name = index_name
    return features


def fix_anomoly(df: pd.DataFrame, col, val):
    df[f'{col}_ANOMOLY'] = df[col] == val
    df[col].replace({val: np.nan}, inplace=True)
