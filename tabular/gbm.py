from ..imports import *

import gc

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from catboost import CatBoostClassifier


def lgb_zero_imp_feats(x: pd.DataFrame, y: np.ndarray, iterations: int = 2) -> Tuple[List, pd.DataFrame]:
    feat_imps = np.zeros(x.shape[1])
    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    for i in range(iterations):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=i)
        model.fit(x_train, y_train, early_stopping_rounds=100, eval_set=[(x_valid, y_valid)], eval_metric='auc',
                  verbose=200)
        feat_imps += model.feature_importances_ / iterations

    feat_imps = pd.DataFrame({'feature': list(x.columns), 'importance': feat_imps})
    zero_features = list(feat_imps[feat_imps['importance'] == 0.0]['feature'])
    return zero_features, feat_imps


def kfold_model(model, df: pd.DataFrame, df_test: pd.DataFrame, id_col: str, y_col: str, n_folds: int = 5,
                rand_seed: int = 777, one_hot: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_ids = df_test[id_col]
    y = df[y_col]
    df = df.drop(columns=[id_col, y_col])
    df_test = df_test.drop(columns=[id_col])

    if one_hot:
        df = pd.get_dummies(df)
        df_test = pd.get_dummies(df_test)
        df, df_test = df.align(df_test, join='inner', axis=1)

    feat_names = list(df.columns)
    x = np.array(df)
    x_test = np.array(df_test)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=rand_seed)
    feat_imp = np.zeros(len(feat_names))
    y_pred = np.zeros(df_test.shape[0])
    y_oof = np.zeros(df.shape[0])

    valid_scores = []
    train_scores = []
    for train_idxs, valid_idxs in k_fold.split(df):
        x_train, y_train = x[train_idxs], y[train_idxs]
        x_valid, y_valid = x[valid_idxs], y[valid_idxs]

        model.create_model()
        model.fit(x_train, y_train, x_valid, y_valid)

        feat_imp += model.get_feat_imp() / n_folds
        y_pred += model.predict_proba(x_test) / n_folds
        y_oof[valid_idxs] = model.predict_proba(x_valid)

        train_score, valid_score = model.get_best_scores()
        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, x_train, x_valid, y_train, y_valid
        gc.collect()

    df_y_pred = pd.DataFrame({id_col: test_ids, y_col: y_pred})
    df_feat_imp = pd.DataFrame({'feature': feat_names, 'importance': feat_imp})

    valid_scores.append(roc_auc_score(y, y_oof))
    train_scores.append(np.mean(train_scores))

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    df_metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

    return df_y_pred, df_feat_imp, df_metrics


class LgbModel:
    def __init__(self):
        self.model = None

    def create_model(self):
        self.model = lgb.LGBMClassifier(n_estimators=10000, objective='binary', class_weight='balanced',
                                        learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, n_jobs=-1)

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model.fit(x_train, y_train, eval_metric='auc', eval_set=[(x_valid, y_valid), (x_train, y_train)],
                       eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=200)

    def get_best_scores(self):
        return self.model.best_score_['learn'][b'AUC'], self.model.best_score_['validation_0'][b'AUC']

    def get_feat_imp(self):
        return self.model.feature_importances_

    def predict_proba(self, x_test):
        best_iteration = self.model.best_iteration_
        return self.model.predict_proba(x_test, num_iteration=best_iteration)[:, 1]


class CatModel:
    def __init__(self, cat_feat_idxs, rand_seed):
        self.model = None
        self.cat_feat_idxs = cat_feat_idxs
        self.rand_seed = rand_seed

    def create_model(self):
        self.model = CatBoostClassifier(custom_loss=['AUC:hints=skip_train~false'], random_seed=self.rand_seed,
                                        use_best_model=True, od_type='Iter', od_wait=100, task_type='GPU')
        self.model.set_params(logging_level='Silent')  # for plotting

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model.fit(x_train, y_train, cat_features=self.cat_feat_idxs, eval_set=(x_valid, y_valid), plot=True)

    def get_best_scores(self):
        return self.model.best_score_['train']['auc'], self.model.best_score_['valid']['auc']

    def get_feat_imp(self):
        return self.model.feature_importances_

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)[:, 1]


def lgb_model(df: pd.DataFrame, df_test: pd.DataFrame, id_col: str, y_col: str, n_folds: int = 5,
              rand_seed: int = 777) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = LgbModel()
    return kfold_model(model, df, df_test, id_col, y_col, n_folds, rand_seed, one_hot=True)


def cat_model(df: pd.DataFrame, df_test: pd.DataFrame, id_col: str, y_col: str, cat_feat_idxs: List[int],
              n_folds: int = 5, rand_seed: int = 777) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = CatModel(cat_feat_idxs, rand_seed)
    return kfold_model(model, df, df_test, id_col, y_col, n_folds, rand_seed, one_hot=False)
