from ..imports import *

from sklearn.model_selection import train_test_split


def missing_val_stats(df: pd.DataFrame) -> pd.DataFrame:
    mis_val = df.isnull().sum()
    mis_val_percent = df.isnull().sum() / len(df)
    df_missing_val = pd.concat([mis_val, mis_val_percent], axis=1)
    df_missing_val = df_missing_val.rename(columns={0: 'Missing Values', 1: 'Fraction'})
    df_missing_val = df_missing_val[df_missing_val.iloc[:, 1] != 0].sort_values('Fraction', ascending=False)
    return df_missing_val


def drop_missing_val(df_train: pd.DataFrame, df_test: pd.DataFrame, pct_threshold: float = 0.9) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    train_miss = missing_val_stats(df_train)
    test_miss = missing_val_stats(df_test)

    missing_train_columns = list(train_miss.index[train_miss['Fraction'] > pct_threshold])
    missing_test_columns = list(test_miss.index[test_miss['Fraction'] > pct_threshold])

    missing_columns = list(set(missing_train_columns + missing_test_columns))
    df_train = df_train.drop(columns=missing_columns)
    df_test = df_test.drop(columns=missing_columns)
    return df_train, df_test


def unique_stats(df: pd.DataFrame) -> pd.DataFrame:
    unique_counts = df.nunique()
    df_unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'n_unique'})
    df_unique_stats = df_unique_stats.sort_values('n_unique', ascending=True)
    return df_unique_stats


def drop_single_unique(df: pd.DataFrame) -> pd.DataFrame:
    unique_counts = df.nunique()
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
        columns={'index': 'feature', 0: 'n_unique'})
    to_drop = list(record_single_unique['feature'])
    return df.drop(columns=to_drop)


def identify_collinear(df: pd.DataFrame, correlation_threshold: float, one_hot: bool = False):
    corr_matrix = pd.get_dummies(df).corr() if one_hot else df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    df_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])
    for column in to_drop:
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]
        temp_df = pd.DataFrame.from_dict(
            {'drop_feature': drop_features, 'corr_feature': corr_features, 'corr_value': corr_values})
        df_collinear = df_collinear.append(temp_df, ignore_index=True)
    return df.drop(columns=to_drop), df_collinear
