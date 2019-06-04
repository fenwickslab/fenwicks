from ..imports import *

import gc
from scipy.stats import kurtosis, iqr, skew


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


def do_mean(df: pd.DataFrame, group_cols: List[str], counted: str, agg_name: str) -> pd.DataFrame:
    gp: pd.DataFrame = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_median(df: pd.DataFrame, group_cols: List[str], counted: str, agg_name: str) -> pd.DataFrame:
    gp: pd.DataFrame = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_std(df: pd.DataFrame, group_cols: List[str], counted: str, agg_name: str) -> pd.DataFrame:
    gp: pd.DataFrame = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_sum(df: pd.DataFrame, group_cols: List[str], counted: str, agg_name: str) -> pd.DataFrame:
    gp: pd.DataFrame = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def label_encoder(df: pd.DataFrame, categorical_columns: List[str] = None, nan_as_category: bool = True) -> Tuple[
    pd.DataFrame, List[str]]:
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


def one_hot_encoder(df: pd.DataFrame, categorical_columns: List[str] = None, nan_as_category: bool = True) -> Tuple[
    pd.DataFrame, List[str]]:
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def group(df_to_agg: pd.DataFrame, prefix: str, aggregations: Dict[str, List[str]],
          aggregate_by: str = 'SK_ID_CURR') -> pd.DataFrame:
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg: pd.DataFrame, df_to_merge: pd.DataFrame, prefix: str, aggregations: Dict[str, List[str]],
                    aggregate_by: str = 'SK_ID_CURR'):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by=aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on=aggregate_by)


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features


def chunk_groups(groupby_object, chunk_size: int):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name: str = 'Index', num_workers: int = 0,
                   chunk_size: int = 100000, num_threads: int = 1) -> pd.DataFrame:
    if num_workers <= 0:
        num_workers = num_threads
    # n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
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
