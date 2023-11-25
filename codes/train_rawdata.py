import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from utils import generate_dic
from utils import evaluation

def model_args(dataset, mdlname):
    n_estimators = 10000
    pos_counts = dataset['trainY'].sum()
    neg_counts = dataset['trainY'].size - pos_counts
    scale_pos_weight = neg_counts/pos_counts

    args1 = {
        'n_estimators': n_estimators,
        'scale_pos_weight': scale_pos_weight,
    }
    args2 = {
        'X': dataset['trainX'],
        'y': dataset['trainY'],
        'verbose': True
    }
    if mdlname=='xgb':
        args1.update({
            'random_state': 0,
            'device': 'cuda',
            'verbosity': 3,
            'objective': 'binary:logistic',
            'learning_rate': 0.03,
            'tree_method': 'hist',
            'enable_categorical': True
        })
        args2.update({
            'eval_set': [(dataset['valX'], dataset['valY'])],
            'eval_metric': ['aucpr'],
            'early_stopping_rounds': 500,
        })
    args = (args1, args2)
    return args

tStart = datetime.now()
__file__ = '/root/ESUN/codes/train_rawdata.py'
prj_folder = os.path.dirname(os.path.dirname(__file__))
path_db = os.path.join(prj_folder,'dataset_1st','db_raw.joblib')
dic = joblib.load(path_db)
dic['pred'].chid = dic['pred'].chid.astype(dic['train'].chid.dtype)
dic['pred'].cano = dic['pred'].cano.astype(dic['train'].cano.dtype)
dataset = generate_dic(path_db)
args = model_args(dataset, 'xgb')
clf = XGBClassifier(**args[0])
clf.fit(**args[1])
tElapsed = datetime.now() - tStart
csv_path = os.path.join(prj_folder, 'train_test', 'evaluation.csv')
evaluation('xgb(raw data)-default hyper-parameters', clf, dataset['testX'], dataset['testY'], csv_path, tElapsed)