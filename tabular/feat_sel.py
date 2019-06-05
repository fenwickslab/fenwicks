from ..imports import *


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    df_missing_val = pd.concat([mis_val, mis_val_percent], axis=1)
    df_missing_val = df_missing_val.rename(columns={0: 'Missing Values', 1: 'Percent'})
    df_missing_val = df_missing_val[df_missing_val.iloc[:, 1] != 0].sort_values('Percent', ascending=False).round(1)
    return df_missing_val


def remove_missing_columns(df_train: pd.DataFrame, df_test: pd.DataFrame, pct_threshold: int = 90) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    train_miss = missing_values_table(df_train)
    test_miss = missing_values_table(df_test)

    missing_train_columns = list(train_miss.index[train_miss['percent'] > pct_threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > pct_threshold])

    missing_columns = list(set(missing_train_columns + missing_test_columns))
    df_train = df_train.drop(columns=missing_columns)
    df_test = df_test.drop(columns=missing_columns)
    return df_train, df_test
