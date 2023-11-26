"""
共用的函式集
"""
import csv
import joblib
import numpy as np
import os
import pandas as pd
import multiprocessing
# import cpuinfo
import re
import json
# import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from socket import gethostname
from catboost import Pool
from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
# from xgboost import DMatrix
# from xgboost import XGBClassifier

def thread_count():
    """
    估算機器使用的CPU運算核心數量上限
    """
    info = cpuinfo.get_cpu_info()
    if info['brand_raw']=='13th Gen Intel(R) Core(TM) i9-13900': # w680
        n_thread = 24-2
    elif info['brand_raw']=='Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz': # yikuan
        n_thread = 6
    else:
        name = re.search('esun\dgpu', gethostname())
        if name is not None: # TWCC container
            n_gpu = int(name[0][4])
            n_thread = n_gpu*4
        else:
            n_thread = multiprocessing.cpu_count()-1
    return n_thread

def parse_catboost(extra, namespace=None):
    """
    catboost model parser
    """
    parser = ArgumentParser()
    # arguments for creating model instance
    parser.add_argument('--n_estimators', type=int, default=50000)
    parser.add_argument('--random_seed', type=int, default=8)
    parser.add_argument('--task_type', type=str, default='GPU')
    parser.add_argument('--bootstrap_type', type=str, default='Poisson')
    parser.add_argument('--target_border', type=float, default=0.5)
    parser.add_argument('--loss_function', type=str, default='Logloss')
    # parser.add_argument('--custom_metric', type=str, default='F1:use_weights=false')
    # parser.add_argument('--eval_metric', type=str, default='F1:use_weights=false')
    parser.add_argument('--custom_metric', type=str, default='PRAUC:type=Classic;use_weights=False')
    parser.add_argument('--eval_metric', type=str, default='PRAUC:type=Classic;use_weights=False')
    # arguments for training
    parser.add_argument('--early_stopping_rounds', type=int, default=500)
    parser.add_argument('--no-early-stopping', dest='early_stopping_rounds', action='store_false')
    args = parser.parse_args(extra, namespace) # add new arguments in orginal args
    return args

def parse_xgboost(extra, namespace=None, gpu_id=0):
    """
    xgboost model parser
    """
    parser = ArgumentParser()
    # arguments for creating model instance
    parser.add_argument('--n_estimators', type=int, default=50000)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--device', type=str, default=f'cuda:{gpu_id}')
    parser.add_argument('--verbosity', type=int, default=3)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--tree_method', type=str, default='hist')
    parser.add_argument('--enable_categorical', action='store_true', default=True)
    parser.add_argument('--disable_categorical', dest='enable_categorical', action='store_false')
    parser.add_argument('--eval_metric', nargs=1, default=['aucpr'])
    parser.add_argument('--early_stopping_rounds', type=int, default=200)
    parser.add_argument('--no-early-stopping', dest='early_stopping_rounds', action='store_const', const=None)
    # arguments for training
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args(extra, namespace)
    return args

def parse_lightgbm(extra, namespace=None):
    """
    lightgbm model parser
    """
    parser = ArgumentParser()
    # arguments for creating model instance
    parser.add_argument('--n_estimators', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--force_row_wise', action='store_true', default=True)
    parser.add_argument('--num_threads', type=int, default=thread_count())
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--colsample_bytree', type=int, default=1)
    parser.add_argument('--reg_lambda', type=float, default=3.0) # L2 regularization
    parser.add_argument('--objective', type=str, default='binary')
    parser.add_argument('--first_metric_only', action='store_true', default=True)
    parser.add_argument('--no-first_metric_only', dest='first_metric_only', action='store_false')
    parser.add_argument('--early_stopping_round', type=int, default=200)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='auc')
    args = parser.parse_args(extra, namespace)
    return args

def catboost_fillna(df_train, df_pred):
    """
    catboost 模型訓練前資料前處理, 補空值
    """
    for col in df_train.columns[df_train.dtypes=='category']:
        if df_train[col].isna().any():
            if issubclass(df_train[col].cat.categories.dtype.type, np.integer):
                minval = np.iinfo(df_train[col].cat.categories.dtype.type).min
                maxval = np.iinfo(df_train[col].cat.categories.dtype.type).max
                if maxval not in df_pred[col].cat.categories:
                    df_train[col] = df_train[col].cat.add_categories(maxval).fillna(maxval)
                    df_pred[col] = df_pred[col].cat.add_categories(maxval).fillna(maxval)
                elif minval not in df_pred[col].cat.categories:
                    df_train[col] = df_train[col].cat.add_categories(minval).fillna(minval)
                    df_pred[col] = df_pred[col].cat.add_categories(minval).fillna(minval)
            elif df_train[col].cat.categories.dtype==object:
                df_train[col] = df_train[col].cat.add_categories('na').fillna('na')
                df_pred[col] = df_pred[col].cat.add_categories('na').fillna('na')
            else:
                raise AssertionError('Unexpect categorical data type')

def catboost_fillna_pred(df_pred):
    """
    catboost 模型預測前資料前處理, 補空值
    """
    for col in df_pred.columns[df_pred.dtypes=='category']:
        if df_pred[col].isna().any():
            if issubclass(df_pred[col].cat.categories.dtype.type, np.integer):
                minval = np.iinfo(df_pred[col].cat.categories.dtype.type).min
                maxval = np.iinfo(df_pred[col].cat.categories.dtype.type).max
                if maxval not in df_pred[col].cat.categories:
                    df_pred[col] = df_pred[col].cat.add_categories(maxval).fillna(maxval)
                elif minval not in df_pred[col].cat.categories:
                    df_pred[col] = df_pred[col].cat.add_categories(minval).fillna(minval)
            elif df_pred[col].cat.categories.dtype==object:
                df_pred[col] = df_pred[col].cat.add_categories('na').fillna('na')
            else:
                raise AssertionError('Unexpect categorical data type')

def ylabel2uint8(df_train, df_pred):
    """
    convert boolean to uint8
    """
    df_train.label = df_train.label.astype(np.uint8)
    df_pred.label = df_pred.label.astype(np.uint8)

def generate_dic(path_db, mdlname, allin=False):
    """
    產生模型訓練所需的dataset
    """
    if isinstance(path_db, str) and os.path.isfile(path_db):
        dic = joblib.load(path_db)
        df_train, df_pred = dic['train'], dic['pred']
    elif isinstance(path_db, dict):
        df_train, df_pred = path_db['train'], path_db['pred']
    else:
        raise AssertionError(f'Unexpect path_db: {path_db}')
    assert df_train.columns.equals(df_pred.columns), 'features mismatch'
    if mdlname=='catboost':
        catboost_fillna(df_train, df_pred)
    if mdlname in {'catboost','lightgbm'}:
        ylabel2uint8(df_train, df_pred)
    columns = df_train.columns.tolist()
    columns.remove('label')
    if allin:
        for col in columns:
            dtype_train = df_train[col].dtype
            dtype_pred = df_pred[col].dtype
            if dtype_pred.name=='category' and dtype_pred!=dtype_train:
                df_train[col] = df_train[col].astype(dtype_pred)
        trainX = pd.concat([df_train[columns], df_pred[columns]], axis=0)
        trainY = pd.concat([df_train.label, df_pred.label], axis=0)
        dic = {
            'trainX': trainX,
            'trainY': trainY
        }
        if mdlname=='catboost':
            dic['trainX_raw'] = dic['trainX']
            cat_features = np.nonzero(((trainX.dtypes==bool) | (trainX.dtypes=='category')).to_numpy())[0]
            dic['trainX'] = Pool(trainX, label=trainY, cat_features=cat_features)
    else:
        trainX = df_train[columns]
        trainY = df_train.label
        testX = df_pred[columns]
        testY = df_pred.label
        dic = {
            'trainX': trainX,
            'trainY': trainY,
            'valX': testX,
            'valY': testY,
            'testX': testX,
            'testY': testY
        }
        if mdlname=='catboost':
            dic['trainX_raw'] = dic['trainX']
            dic['testX_raw'] = dic['testX']
            cat_features = np.nonzero(((trainX.dtypes==bool) | (trainX.dtypes=='category')).to_numpy())[0]
            dic['trainX'] = Pool(trainX, label=trainY, cat_features=cat_features)
            dic['testX'] = Pool(testX, label=testY, cat_features=cat_features)
            dic['valX'] = dic['testX']
    return dic

def model_args(dataset, mdlname, args_model):
    """
    產生模型訓練所需的input arguments
    """
    args0 = vars(args_model) # convert to dictionary
    if mdlname=='catboost':
        no_early_stopping = isinstance(args_model.early_stopping_rounds, bool)
        args1 = dict()
        if no_early_stopping: # no-early-stopping
            args0.pop('early_stopping_rounds')
        else:
            args1['early_stopping_rounds'] = args0.pop('early_stopping_rounds')
        # training set
        if isinstance(dataset['trainX'], pd.DataFrame):
            args1['X'] = dataset['trainX']
            args1['y'] = dataset['trainY']
        elif isinstance(dataset['trainX'], Pool):
            args1['X'] = dataset['trainX']
        else:
            raise AssertionError('Unexpect data format of trainX')
        # validation set
        if not no_early_stopping and 'valX' in dataset:
            if isinstance(dataset['valX'], pd.DataFrame):
                args1['eval_set'] = (dataset['valX'], dataset['valY'])
            elif isinstance(dataset['valX'], Pool):
                args1['eval_set'] = dataset['valX']
            else:
                raise AssertionError('Unexpect data format of valX')
    elif mdlname=='xgboost':
        args1 = {'verbose': args0.pop('verbose')}
        # training set
        args1['X'] = dataset['trainX']
        args1['y'] = dataset['trainY']
        # validation set
        if args_model.early_stopping_rounds is not None:
            args1['eval_set'] = [(dataset['valX'], dataset['valY'])]
    # deal with imbalanced distribution
    pos_counts = dataset['trainY'].sum()
    neg_counts = dataset['trainY'].size - pos_counts
    args0['scale_pos_weight'] = neg_counts/pos_counts
    return (args0, args1)

def print_input(args_input, args_mdinput, param):
    """
    print input 內容
    """
    print('[Input Parameters]')
    rm_names = {'gpu_id','params'}
    d = dict()
    for k,v in vars(args_input).items():
        if k not in rm_names:
            d[k] = v
    print(json.dumps(d, indent=4))
    print('[Model Parameters]')
    rm_names = {'X','y','eval_set'}
    if 'n_estimators' in args_input:
        rm_names.add('n_estimators')
    d = dict()
    for k,v in args_mdinput[0].items():
        if k not in rm_names:
            d[k] = v
    for k,v in args_mdinput[1].items():
        if k not in rm_names:
            d[k] = v
    print(json.dumps(d, indent=4))
    print('[Model Parameters (User Assigned)]')
    print(json.dumps(param, indent=4))

def dump_opt_thres(estimator, dataset, json_path, append_info=dict()):
    """
    輸出最佳threshold
    """
    X = dataset['trainX']
    y_true = dataset['trainY'].to_numpy()
    y_score = estimator.predict_proba(X)
    precision, recall, threshold = precision_recall_curve(y_true, y_score[:,1])
    f1 = 2*precision*recall/(precision+recall)
    ix = np.nanargmax(f1)
    info = {'opt_f1': f1[ix], 'opt_thres': threshold[ix]}
    info.update(append_info)
    with open(json_path, 'wt') as f:
        json.dump(info, f, indent=4)

def evaluation(mdlfullname, estimator, dataset, csv_path, dt):
    """
    模型訓練好後進行評估, 計算metrics
    """
    X = dataset['trainX']
    y_true = dataset['trainY'].to_numpy()
    y_score = estimator.predict_proba(X)
    # select optimal threshold
    precision, recall, threshold = precision_recall_curve(y_true, y_score[:,1])
    ix = np.nanargmax(2*precision*recall/(precision+recall))
    opt_thres = threshold[ix]
    # apply on testing set
    X = dataset['testX']
    y_true = dataset['testY'].to_numpy() # convert pandas.Series to numpy array
    y_score = estimator.predict_proba(X)
    # area under (ROC, Precision-recall curve)
    roc_auc = roc_auc_score(y_true, y_score[:,1])
    precision, recall, _ = precision_recall_curve(y_true, y_score[:,1])
    pr_auc = auc(recall, precision)
    # model predictions (default threshold)
    y_pred = (y_score[:,1] > 0.5).astype(np.uint8)
    C = confusion_matrix(y_true,y_pred)
    TN = C[0,0]; FP = C[0,1]
    FN = C[1,0]; TP = C[1,1]
    precision = TP/(TP+FP)
    NPV = TN/(TN+FN) # negative predictive value
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = 2*precision*recall/(precision+recall)
    f05 = 1.25*precision*recall/(0.25*precision+recall)
    f2 = 5*precision*recall/(4*precision+recall)
    # model prediction (optimal threshold)
    y_pred_opt = (y_score[:,1] > opt_thres).astype(np.uint8)
    C = confusion_matrix(y_true,y_pred_opt)
    TN = C[0,0]; FP = C[0,1]
    FN = C[1,0]; TP = C[1,1]
    opt_precision = TP/(TP+FP)
    opt_NPV = TN/(TN+FN)
    opt_recall = TP/(TP+FN)
    opt_specificity = TN/(TN+FP)
    opt_accuracy = (TP+TN)/(TP+TN+FP+FN)
    opt_f1 = 2*opt_precision*opt_recall/(opt_precision+opt_recall)
    opt_f05 = 1.25*opt_precision*opt_recall/(0.25*opt_precision+opt_recall)
    opt_f2 = 5*opt_precision*opt_recall/(4*opt_precision+opt_recall)
    print(
        f'\n        *****************************************\n\
         Precision: {precision} \n\
         NPV: {NPV} \n\
         Sensitivity(Recall): {recall} \n\
         Specificity: {specificity} \n\
         Accuracy: {accuracy} \n\
         F0.5: {f05} \n\
         F1: {f1} \n\
         F2: {f2} \n\
         PR_AUC: {pr_auc} \n\
         ROC_AUC: {roc_auc} \n\
         ----------------------------------------- \n\
         opt_Threshold: {opt_thres} \n\
         opt_Precision: {opt_precision} \n\
         opt_NPV: {opt_NPV} \n\
         opt_Sensitivity(Recall): {opt_recall} \n\
         opt_Specificity: {opt_specificity} \n\
         opt_Accuracy: {opt_accuracy} \n\
         opt_F0.5: {opt_f05} \n\
         opt_F1: {opt_f1} \n\
         opt_F2: {opt_f2} \n\
        *****************************************\n'
    )
    # write to csv file
    if not os.path.isfile(csv_path):
        output = ['HostName','Time','Model','Best Iteration','feature number','Cost time',
                  'Precision','NPV','Recall','Specificity','Accuracy','F0.5','F1','F2','PR_AUC','ROC_AUC',
                  'opt_Threshold', 'opt_Precision','opt_NPV','opt_Recall','opt_Specificity','opt_Accuracy','opt_F0.5','opt_F1','opt_F2']
        with open(csv_path, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(output)
    # get best iteration
    if isinstance(estimator, CatBoostClassifier):
        best_iteration = estimator.best_iteration_
        if best_iteration is None: # early stopping is not set
            best_iteration = estimator.get_param('n_estimators')
            if best_iteration is None:
                best_iteration = estimator.get_param('iterations')
                if best_iteration is None:
                    best_iteration = estimator.get_param('num_trees')
                    if best_iteration is None:
                        best_iteration = estimator.get_param('num_boost_round')
            if best_iteration is not None:
                best_iteration -= 1
    elif isinstance(estimator, XGBClassifier):
        if estimator.early_stopping_rounds is None:
            best_iteration = estimator.n_estimators
        else:
            best_iteration = estimator.best_iteration
    else:
        best_iteration = 'Not support'
    output = [gethostname(), datetime.now(), mdlfullname, best_iteration, X.shape[1], dt,
              precision, NPV, recall, specificity, accuracy, f05, f1, f2, pr_auc, roc_auc,
              opt_thres, opt_precision, opt_NPV, opt_recall, opt_specificity, opt_accuracy, opt_f05, opt_f1, opt_f2]
    with open(csv_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(output)
    return f1, pr_auc, opt_thres, opt_f1

def search_results(f1, pr_auc, opt_thres, opt_f1, params, csv_path):
    """
    輸出Grid search的結果到csv檔
    """
    if not os.path.isfile(csv_path):
        output = list(params) + ['F1','PR_AUC','opt_thres','opt_F1']
        with open(csv_path, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(output)
    output = list(params.values()) + [f1, pr_auc, opt_thres, opt_f1]
    with open(csv_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(output)

# def model_training(db_path,mdlname,gpu_id,mdl_prior,iter,path_prior,xls_path=None,feature_history=None,pre_features=None):
#     dataset = generate_dic(db_path)
#     args = model_args(dataset,mdlname,gpu_id)

#     if iter in ('all','final'):
#         mdlfullname = '{}-{}'.format(mdl_prior, iter)
#     else:
#         mdlfullname = '{}-iter{:02}'.format(mdl_prior, iter)
#     print(f'[Fitting Model]: {mdlfullname}')
#     t_start = datetime.now()
#     # mdl_path = os.path.join(path_prior, mdlfullname)
#     if mdlname=='xgb':
#         model = XGBClassifier(**args[0])
#         model.fit(**args[1])
#         # model.save_model(mdl_path+'.xgb')
#     elif mdlname=='catboost':
#         model = CatBoostClassifier(**args[0])
#         model.fit(**args[1])
#         # model.save_model(mdl_path+'.cbm')
#     elif mdlname=='lightgbm':
#         model = LGBMClassifier(**args[0])
#         model.fit(**args[1])
#         # joblib.dump(model, mdl_path+'.joblib', compress=3, protocol=4)

#     dt = datetime.now() - t_start
#     print(f'[Evaluation Model and Output]: {mdlfullname}')
#     csv_path = os.path.join(path_prior, f'{mdlname}.csv')
#     evaluation(mdlfullname, model, dataset['testX'], dataset['testY'], csv_path, dt)
