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

    tf.logging.INFO(f'correlation between {var_name} and the target is {corr:0.4f}')
    tf.logging.INFO(f'Median value for positive target: {df.ix[df[target_name] == 1, var_name].median():0.4f}')
    tf.logging.INFO(f'Median value for negative target: {df.ix[df[target_name] == 0, var_name].median():0.4f}')
