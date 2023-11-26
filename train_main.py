"""
模型訓練主程式
"""
# Syntax
#       python train_main.py [db_name] [mdlname] [param1] [param2] ... [paramN] [--combineTrainTest] [--no-early-stopping] [--n_estimators n1 n2 ... nN]
import os
import joblib
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from catboost import Pool
from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from datetime import datetime
from utils import parse_catboost
from utils import parse_xgboost
from utils import generate_dic
from utils import model_args
from utils import print_input
from utils import dump_opt_thres
from utils import evaluation
from codes.shap_importance import catboost_shap
from codes.shap_importance import xgb_shap as xgboost_shap

def parse_args():
    """
    parse input
    """
    parser = ArgumentParser()
    parser.add_argument('db_name', type=str)
    parser.add_argument('mdlname', type=str)
    parser.add_argument('params', nargs='+') # at least one param should be given
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.add_argument('--calc_shap', action='store_true', default=True)
    parser.add_argument('--no-calc_shap', dest='calc_shap', action='store_false')
    parser.add_argument('--batch_size', type=int, default=100000) # batch processing to prevent out-of-memory
    parser.add_argument('--combineTrainTest', action='store_true', default=False)
    parser.add_argument('--n_estimators', nargs='*', type=int, default=[10000]) # override model's input arguments
    args_input, extra = parser.parse_known_args()
    if args_input.combineTrainTest and '--no-early-stopping' not in extra:
        extra.append('--no-early-stopping')
    if args_input.mdlname=='catboost':
        args_model = parse_catboost(extra)
    elif args_input.mdlname=='xgboost':
        args_model = parse_xgboost(extra, None, args_input.gpu_id)
    else:
        raise AssertionError(f'Unexpect model name: {args_input.mdlname}')
    n_param = len(args_input.params)
    n_estimator = len(args_input.n_estimators)
    if n_param > 1 and n_estimator==1:
        args_input.n_estimators = args_input.n_estimators*n_param
        n_estimator = n_param
    assert n_estimator==n_param, 'number of n_estimator does not match number of params'
    return args_input, args_model


if __name__=='__main__':
    args_input, args_model = parse_args()

    # __file__ = '/root/ESUN/codes/train_main.py'
    prj_folder = os.path.dirname(__file__)
    if int(args_input.db_name[4]) < 4:
        path_db = os.path.join(prj_folder, 'dataset_1st', args_input.db_name+'.joblib')
    else:
        path_db = os.path.join(prj_folder, 'dataset_2nd', args_input.db_name+'.joblib')
    mdl_folder = os.path.join(prj_folder,'train_test',f'{args_input.mdlname}_search')
    if not os.path.isdir(mdl_folder):
        os.makedirs(mdl_folder)

    # load dataset
    allin = args_input.combineTrainTest
    dataset = generate_dic(path_db, args_input.mdlname, allin=allin)
    args_mdinput = model_args(dataset, args_input.mdlname, args_model)
    args_mdinput[0]['scale_pos_weight'] = 1 # 經測試，設為1會有較好的F1 score

    if args_input.mdlname=='catboost':
        param_names = ['max_depth','learning_rate','max_leaves','subsample','reg_lambda','min_data_in_leaf']
        trainX = dataset['trainX_raw']
        if not allin:
            testX = dataset['testX_raw']
    elif args_input.mdlname=='xgboost':
        param_names = ['max_depth','learning_rate','min_child_weight','gamma','subsample','colsample_bytree']
        trainX = dataset['trainX']
        testX = dataset['testX']

    if args_input.calc_shap:
        if allin:
            y_true = dataset['trainY'].to_numpy()
            if args_input.mdlname=='catboost':
                X = dataset['trainX_raw']
                poolX = dataset['trainX']
            else:
                X = dataset['trainX']
        else:
            for col in trainX.columns:
                dtype_train = trainX[col].dtype
                dtype_test = testX[col].dtype
                if dtype_test.name=='category' and dtype_test!=dtype_train:
                    trainX[col] = trainX[col].astype(dtype_test)
            X = pd.concat([trainX, testX], axis=0)
            y_true = np.r_[dataset['trainY'].to_numpy(), dataset['testY'].to_numpy()]
            if args_input.mdlname=='catboost':
                cat_features = np.nonzero(((X.dtypes==bool)|(X.dtypes=='category')).to_numpy())[0]
                poolX = Pool(X, label=y_true, cat_features=cat_features)

    n_param = len(args_input.params)
    append_info = {'db_name': args_input.db_name, 'mdlname': args_input.mdlname}
    for p in range(n_param):
        print(f'[{p+1}/{n_param}] catboost:{args_input.params[p]}')
        values = args_input.params[p].split('_')[::-1]
        param = {k:eval(v) for k,v in zip(param_names, values)}
        # overwrite n_estimators
        args_mdinput[0]['n_estimators'] = args_input.n_estimators[p]
        print_input(args_input, args_mdinput, param)
        mdlfile = f'{args_input.db_name}-{args_input.params[p]}'
        if allin:
            mdlfile += '(allin)'
        csv_path = os.path.join(mdl_folder, mdlfile+'.csv')
        if args_input.mdlname=='catboost':
            mdl_path = os.path.join(mdl_folder, mdlfile+'.cbm')
            append_info['mdlfile'] = [mdlfile+'.cbm']
            print(' >>> start training <<<')
            model = CatBoostClassifier(**args_mdinput[0], **param)
            tStart = datetime.now()
            model.fit(**args_mdinput[1])
            tElapsed = datetime.now() - tStart
            if allin:
                json_path = os.path.join(mdl_folder, mdlfile+'_opt_thres.json')
                dump_opt_thres(model, dataset, json_path, append_info)
            else:
                f1, pr_auc, opt_thres, opt_f1 = evaluation(args_input.params[p], model, dataset, csv_path, tElapsed)
            if args_input.save_model:
                model.save_model(mdl_path, format='cbm')

        elif args_input.mdlname=='xgboost':
            mdl_path = os.path.join(mdl_folder, mdlfile+'.joblib')
            append_info['mdlfile'] = [mdlfile+'.joblib']
            print(' >>> start training <<<')
            model = XGBClassifier(**args_mdinput[0], **param)
            tStart = datetime.now()
            model.fit(**args_mdinput[1])
            tElapsed = datetime.now() - tStart
            if allin:
                json_path = os.path.join(mdl_folder, mdlfile+'_opt_thres.json')
                dump_opt_thres(model, dataset, json_path, append_info)
            else:
                f1, pr_auc, opt_thres, opt_f1 = evaluation(args_input.params[p], model, dataset, csv_path, tElapsed)
            if args_input.save_model:
                joblib.dump(model, mdl_path, compress=3, protocol=4)


        if args_input.calc_shap:
            if args_input.mdlname=='catboost':
                y_pred = model.predict(poolX)
            elif args_input.mdlname=='xgboost':
                y_pred = model.predict(X)
            lg_correct = y_true==y_pred
            lg_positive = y_true > 0
            lg_negative = ~lg_positive
            ix_pos = np.nonzero(lg_correct & lg_positive)[0]
            ix_neg = np.nonzero(lg_correct & lg_negative)[0]
            if args_input.mdlname=='catboost':
                shap_values_pos, shap_values_neg = catboost_shap(model, poolX, ix_pos, ix_neg, args_input.batch_size)
            elif args_input.mdlname=='xgboost':
                shap_values_pos, shap_values_neg = xgboost_shap(model, X, ix_pos, ix_neg, args_input.batch_size)

            xls_path = os.path.join(mdl_folder, mdlfile+'_shap.xlsx')
            with pd.ExcelWriter(xls_path) as writer:
                pd.Series(shap_values_pos, index=X.columns).sort_values(ascending=False).to_excel(writer, sheet_name='Positive')
                pd.Series(shap_values_neg, index=X.columns).sort_values(ascending=False).to_excel(writer, sheet_name='Negative')

        tElapsed = datetime.now() - tStart
        print('Elapsed time:', tElapsed)
