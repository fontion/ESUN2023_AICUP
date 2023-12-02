"""
計算模型預測時每個feature的貢獻度
"""

# Syntax
#       python shap_importance.py [db_name] [mdlname] [params]

import joblib
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from catboost import Pool
from catboost import CatBoostClassifier
# from xgboost import DMatrix
from utils import catboost_fillna
from utils import ylabel2uint8
from tqdm import tqdm
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import auc
# from sklearn.metrics import confusion_matrix


def parse_args():
    """
    parse input
    """
    parser = ArgumentParser()
    parser.add_argument('db_name', type=str)
    parser.add_argument('mdlname', type=str)
    parser.add_argument('params', nargs='+') # at least one param should be given
    parser.add_argument('--eval_metric', type=str, default='F1') # use to identify model path
    parser.add_argument('--subfolder', type=str, default='scale_pos_weight=1')
    parser.add_argument('--batch_size', type=int, default=0) # Do not batch processing by default(batch_size=0)
    return parser.parse_args()

def load_db(path_db, mdlname):
    """
    載入database並做適當前處理供後續訓練
    """
    df_train, df_pred = joblib.load(path_db).values()
    if mdlname=='catboost':
        catboost_fillna(df_train, df_pred)
    if mdlname in {'catboost','lightgbm'}:
        ylabel2uint8(df_train, df_pred)
    columns = df_train.columns.tolist()
    columns.remove('label')
    for col in columns:
        dtype_train = df_train[col].dtype
        dtype_pred = df_pred[col].dtype
        if dtype_pred.name=='category' and dtype_pred!=dtype_train:
            df_train[col] = df_train[col].astype(dtype_pred)
    X = pd.concat([df_train[columns], df_pred[columns]], axis=0)
    y = pd.concat([df_train.label, df_pred.label], axis=0)
    return X, y.to_numpy()

def split_task(batch_size, n_sample):
    """
    計算批次處理所需的index
    """
    batch_idx = list(range(0, n_sample, batch_size))
    batch_idx.append(n_sample)
    return batch_idx

def catboost_shap(model, X, ix_pos, ix_neg, batch_size=0):
    """
    計算catboost模型預測時每個feature的shapley value
    """
    n_pos = ix_pos.size
    n_neg = ix_neg.size
    if batch_size > 0:
        shap_values_pos = np.zeros(X.shape[1])
        shap_values_neg = np.zeros(X.shape[1])
        batch_idx_pos = split_task(batch_size, n_pos)
        batch_idx_neg = split_task(batch_size, n_neg)
        if isinstance(X, Pool):
            for b in tqdm(range(len(batch_idx_pos)-1), desc='positive'):
                sli = slice(batch_idx_pos[b], batch_idx_pos[b+1])
                pool = X.slice(ix_pos[sli])
                shap_values = model.get_feature_importance(pool, type='ShapValues')
                shap_values_pos += np.abs(shap_values[:,:-1]).sum(axis=0)
            for b in tqdm(range(len(batch_idx_neg)-1), desc='negative'):
                sli = slice(batch_idx_neg[b], batch_idx_neg[b+1])
                pool = X.slice(ix_neg[sli])
                shap_values = model.get_feature_importance(pool, type='ShapValues')
                shap_values_neg += np.abs(shap_values[:,:-1]).sum(axis=0)
        elif isinstance(X, pd.DataFrame):
            cat_features = np.nonzero(((X.dtypes==bool)|(X.dtypes=='category')).to_numpy())[0]
            for b in tqdm(range(len(batch_idx_pos)-1), desc='positive'):
                sli = slice(batch_idx_pos[b], batch_idx_pos[b+1])
                pool = Pool(X.iloc[ix_pos[sli]], cat_features=cat_features)
                shap_values = model.get_feature_importance(pool, type='ShapValues')
                shap_values_pos += np.abs(shap_values[:,:-1]).sum(axis=0)
            for b in tqdm(range(len(batch_idx_neg)-1), desc='negative'):
                sli = slice(batch_idx_neg[b], batch_idx_neg[b+1])
                pool = Pool(X.iloc[ix_neg[sli]], cat_features=cat_features)
                shap_values = model.get_feature_importance(pool, type='ShapValues')
                shap_values_neg += np.abs(shap_values[:,:-1]).sum(axis=0)
    else:
        if isinstance(X, Pool):
            pool_pos = X.slice(ix_pos)
            pool_neg = X.slice(ix_neg)
        elif isinstance(X, pd.DataFrame):
            cat_features = np.nonzero(((X.dtypes==bool)|(X.dtypes=='category')).to_numpy())[0]
            print('Building positive sample pool...')
            pool_pos = Pool(X.iloc[ix_pos], cat_features=cat_features)
            print('Building negative sample pool...')
            pool_neg = Pool(X.iloc[ix_neg], cat_features=cat_features)
        shap_values_pos = model.get_feature_importance(pool_pos, type='ShapValues')
        shap_values_pos = np.abs(shap_values_pos[:,:-1]).sum(axis=0)
        shap_values_neg = model.get_feature_importance(pool_neg, type='ShapValues')
        shap_values_neg = np.abs(shap_values_neg[:,:-1]).sum(axis=0)
    shap_values_pos /= n_pos
    shap_values_neg /= n_neg
    return shap_values_pos, shap_values_neg

def xgb_shap(model, X, ix_pos, ix_neg, batch_size):
    """
    計算xgboost模型預測時每個feature的shapley value
    """
    n_pos = ix_pos.size
    n_neg = ix_neg.size
    booster = model.get_booster()
    booster.set_param({"device": "cuda:0"})
    if batch_size > 0:
        shap_values_pos = np.zeros(X.shape[1])
        batch_idx = split_task(batch_size, n_pos)
        for b in tqdm(range(len(batch_idx)-1), desc='positive'):
            sli = slice(batch_idx[b], batch_idx[b+1])
            dtrain = DMatrix(X.iloc[ix_pos[sli]], enable_categorical=True)
            shap_values = booster.predict(dtrain, pred_contribs=True)
            shap_values_pos += np.abs(shap_values[:,:-1]).sum(axis=0)
        shap_values_neg = np.zeros(X.shape[1])
        batch_idx = split_task(batch_size, n_neg)
        for b in tqdm(range(len(batch_idx)-1), desc='negative'):
            sli = slice(batch_idx[b], batch_idx[b+1])
            dtrain = DMatrix(X.iloc[ix_neg[sli]], enable_categorical=True)
            shap_values = booster.predict(dtrain, pred_contribs=True)
            shap_values_neg += np.abs(shap_values[:,:-1]).sum(axis=0)
    else:
        print('Building positive sample DMatrix...')
        dtrain = DMatrix(X.iloc[ix_pos], enable_categorical=True)
        shap_values_pos = booster.predict(dtrain, pred_contribs=True)
        shap_values_pos = np.abs(shap_values_pos[:,:-1]).sum(axis=0)
        print('Building negative sample DMatrix...')
        dtrain = DMatrix(X.iloc[ix_neg], enable_categorical=True)
        shap_values_neg = booster.predict(dtrain, pred_contribs=True)
        shap_values_neg = np.abs(shap_values_neg[:,:-1]).sum(axis=0)
    shap_values_pos /= n_pos
    shap_values_neg /= n_neg
    return shap_values_pos, shap_values_neg


if __name__=='__main__':
    args = parse_args()
    print('Model:', args.mdlname)
    print('Loading database:', args.db_name)
    # __file__ = '/root/ESUN/codes/shap_importance.py'
    prj_folder = os.path.dirname(os.path.dirname(__file__))
    path_db = os.path.join(prj_folder,'dataset_2nd',f'{args.db_name}.joblib')
    X, y_true = load_db(path_db, args.mdlname)

    mdl_folder = os.path.join(prj_folder,'train_test',f'{args.mdlname}_search',
                              f'eval_metric={args.eval_metric}',args.db_name[3:5],args.subfolder)
    assert os.path.isdir(mdl_folder), f'Model folder not exist: {mdl_folder}'
    for param in args.params:
        print('Processing:', param)
        if args.mdlname=='catboost':
            mdlfile = f'{args.db_name}-{param}(allin).cbm'
            path_mdl = os.path.join(mdl_folder, mdlfile)
            if not os.path.isfile(path_mdl):
                raise AssertionError(f'Model file not found: {path_mdl}')
            model = CatBoostClassifier()
            model.load_model(path_mdl)
        elif args.mdlname=='xgboost':
            mdlfile = f'{args.db_name}-{param}.joblib'
            path_mdl = os.path.join(mdl_folder, mdlfile)
            if not os.path.isfile(path_mdl):
                raise AssertionError(f'Model file not found: {path_mdl}')
            model = joblib.load(path_mdl)
        y_pred = model.predict(X)
        lg_correct = y_true==y_pred
        lg_positive = y_true > 0
        lg_negative = ~lg_positive
        ix_pos = np.nonzero(lg_correct & lg_positive)[0]
        ix_neg = np.nonzero(lg_correct & lg_negative)[0]
        if args.mdlname=='catboost':
            shap_values_pos, shap_values_neg = catboost_shap(model, X, ix_pos, ix_neg, args.batch_size)
        elif args.mdlname=='xgboost':
            shap_values_pos, shap_values_neg = xgb_shap(model, X, ix_pos, ix_neg, args.batch_size)

        xls_path = os.path.join(mdl_folder, os.path.splitext(mdlfile)[0]+'_shap.xlsx')
        with pd.ExcelWriter(xls_path) as writer:
            pd.Series(shap_values_pos, index=X.columns).sort_values(ascending=False).to_excel(writer, sheet_name='Positive')
            pd.Series(shap_values_neg, index=X.columns).sort_values(ascending=False).to_excel(writer, sheet_name='Negative')
