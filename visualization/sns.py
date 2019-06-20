from ..imports import *

import matplotlib.pyplot as plt
import seaborn as sns


def kde_binary_target(df: pd.DataFrame, var_name: str, target_name: str = 'TARGET'):
    corr = df[target_name].corr(df[var_name])

    plt.figure(figsize=(12, 6))
    sns.kdeplot(df.ix[df[target_name] == 0, var_name], label='TARGET == 0')
    sns.kdeplot(df.ix[df[target_name] == 1, var_name], label='TARGET == 1')

    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()
    plt.show()

    logging.INFO(f'correlation between {var_name} and the target is {corr:0.4f}')
    logging.INFO(f'Median value for positive target: {df.ix[df[target_name] == 1, var_name].median():0.4f}')
    logging.INFO(f'Median value for negative target: {df.ix[df[target_name] == 0, var_name].median():0.4f}')


def plot_collinear(corr_matrix, df_collinear: pd.DataFrame = None):
    """
    Heatmap of the correlation values. When `df_collinear` is provided, the features on the x-axis are those that will
    be removed. The features on the y-axis are the correlated features with those on the x-axis.
    """

    if df_collinear:
        corr_matrix = corr_matrix.loc[list(set(df_collinear['corr_feature'])), list(set(df_collinear['drop_feature']))]

    f, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, center=0, linewidths=.25, cbar_kws={"shrink": 0.6})
    ax.set_yticks([x + 0.5 for x in list(range(corr_matrix.shape[0]))])
    ax.set_yticklabels(list(corr_matrix.index), size=int(160 / corr_matrix.shape[0]))
    ax.set_xticks([x + 0.5 for x in list(range(corr_matrix.shape[1]))])
    ax.set_xticklabels(list(corr_matrix.columns), size=int(160 / corr_matrix.shape[1]))
    plt.title('Correlations', size=14)
