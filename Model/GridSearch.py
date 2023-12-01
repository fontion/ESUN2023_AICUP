"""
搜尋模型訓練時的最佳參數
"""
# Syntax
#       python GridSearch.py [db_name] [xgboost/catboost] [a] [b] [c] [d] [e] [f] [--gpu_id=0] [--save_model_thres=1.78] [eval_metric=F1:use_weights=False] ...
# import logging
import os
import joblib
from argparse import ArgumentParser
from datetime import datetime
# from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import Pool
from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from utils import parse_catboost
from utils import parse_xgboost
from utils import generate_dic
from utils import model_args
from utils import print_input
from utils import evaluation
from utils import search_results

def parse_args():
    """
    parse input
    """
    parser = ArgumentParser()
    parser.add_argument('db_name', type=str)
    parser.add_argument('mdlname', type=str)
    parser.add_argument('params', nargs='*', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--pos_weight_1', action='store_true', default=True)
    parser.add_argument('--no-pos_weight_1', dest='pos_weight_1', action='store_false')
    parser.add_argument('--scale_pos_weight', type=float, default=None)
    args_input, extra = parser.parse_known_args()
    if 'FirstUse' in args_input.db_name:
        save_model_thres = 1.803
    elif 'UsedBefore' in args_input.db_name:
        save_model_thres = 1.72
    else:
        save_model_thres = 1.76
    parser = ArgumentParser()
    parser.add_argument('--save_model_thres', type=float, default=save_model_thres) # threshold for saving model. (value = PR_AUC + opt_F1)
    args_input, extra = parser.parse_known_args(extra, args_input)
    if args_input.mdlname=='catboost':
        args_model = parse_catboost(extra)
    elif args_input.mdlname=='xgboost':
        args_model = parse_xgboost(extra, None, args_input.gpu_id)
    else:
        raise AssertionError(f'Unexpect model name: {args_input.mdlname}')
    # post-process params
    args_input.params = [eval(p) for p in args_input.params]
    n_param = len(args_input.params)
    if n_param < 6: # 目前最多是xgboost search 6個參數，如果有再增加要調整
        args_input.params += [None]*(6-n_param)
    return args_input, args_model

def model_search_params(mdlname):
    """
    parameters for model search
    """
    if mdlname=='catboost':
        params = {
            'max_depth': [2,3,6,7,8],
            'learning_rate': [0.02, 0.03, 0.04, 0.8],
            'max_leaves': [31],
            'subsample': [0.66, 0.7, 0.75, 0.8],
            'reg_lambda': [3, 3.5, 2.5, 2],
            'min_data_in_leaf': [1]
        } # total search space: 5 x 3 x 1 x 4 x 2 = 120
    elif mdlname=='xgboost':
        params = {
            'max_depth': [12, 10, 8, 6],
            'learning_rate': [0.007, 0.01, 0.03, 0.1],
            'min_child_weight': [1,2,5],
            'gamma': [0, 0.5, 1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            # 'alpha': [0, 1, 10],
            # 'lambda': [0, 1, 10],
        } # total search space: 4 x 4 x 3 x 3 x 3 x 3 = 1296
    elif mdlname=='lightgbm':
        params = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.1, 0.01, 0.005],
            'num_leaves': [31, 25, 37],
            'min_data_in_leaf': [20, 15, 25],
        }
    else:
        raise AssertionError(f'Unexpect model name: {mdlname}')
    return params

# def lgb_f1_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.where(y_hat < 0.5, 0, 1)  # scikits f1 doesn't like probabilities
#     return 'f1', f1_score(y_true, y_hat), True

if __name__=='__main__':
    # logging.basicConfig(filename='GridSearch.log', encoding='utf-8', level='logging.DEBUG', format='%(levelname)s:%(asctime)s %(message)s')
    args_input, args_model = parse_args()
    db_name = args_input.db_name
    mdlname = args_input.mdlname
    a_range, b_range, c_range, d_range, e_range, f_range = args_input.params
    print('Search model:', mdlname)
    print('Loading database:', db_name)
    # __file__ = '/root/ESUN/codes/GridSearch.py'
    prj_folder = os.path.dirname(os.path.dirname(__file__))
    if int(db_name[4])<4:
        path_db = os.path.join(prj_folder,'dataset_1st',f'{db_name}.joblib')
    else:
        path_db = os.path.join(prj_folder,'dataset_2nd',f'{db_name}.joblib')
    params = model_search_params(mdlname)
    if mdlname=='catboost' and all((p is not None for p in args_input.params)):
        print('Search parameters:','{}_{}_{}_{}_{}_{}'.format(*[v[i] for v,i in zip(params.values(), args_input.params)][::-1]))
    elif mdlname=='xgboost' and all((p is not None for p in args_input.params)):
        print('Search parameters:','{}_{}_{}_{}_{}_{}'.format(*[v[i] for v,i in zip(params.values(), args_input.params)][::-1]))
    dataset = generate_dic(path_db, mdlname)
    args_mdinput = model_args(dataset, mdlname, args_model)
    if args_input.pos_weight_1:
        args_mdinput[0]['scale_pos_weight'] = 1 # 經測試，設為1會有較好的F1 score
    if args_input.scale_pos_weight is not None:
        args_mdinput[0]['scale_pos_weight'] = args_input.scale_pos_weight
    print('eval_metric:', args_mdinput[0]['eval_metric'])
    if mdlname=='xgboost':
        print('device =', args_mdinput[0]['device'])
    elif mdlname=='catboost':
        print('task_type =', args_mdinput[0]['task_type'])
    csv_path1 = os.path.join(prj_folder, 'train_test', f'evaluation_{mdlname}.csv')
    mdl_folder = os.path.join(prj_folder, 'train_test', f'{mdlname}_search')
    csv_path2 = os.path.join(mdl_folder, f'{db_name}_search.csv')
    if not os.path.isdir(mdl_folder):
        os.makedirs(mdl_folder)

    if mdlname=='xgboost':
        f_range = [f_range] if f_range is not None else range(len(params['colsample_bytree']))
        e_range = [e_range] if e_range is not None else range(len(params['subsample']))
        d_range = [d_range] if d_range is not None else range(len(params['gamma']))
        c_range = [c_range] if c_range is not None else range(len(params['min_child_weight']))
        b_range = [b_range] if b_range is not None else range(len(params['learning_rate']))
        a_range = [a_range] if a_range is not None else range(len(params['max_depth']))
        for f in f_range:
            for e in e_range:
                for d in d_range:
                    for c in c_range:
                        for b in b_range:
                            for a in a_range:
                                hyperparam = '{}_{}_{}_{}_{}_{}'.format(params['colsample_bytree'][f], params['subsample'][e], params['gamma'][d], params['min_child_weight'][c], params['learning_rate'][b], params['max_depth'][a])
                                mdl_path = os.path.join(mdl_folder, f'{db_name}-{hyperparam}.joblib')
                                if os.path.isfile(mdl_path):
                                    print('{} exist, skip current condition'.format(os.path.basename(mdl_path)))
                                else:
                                    param = {
                                        'max_depth': params['max_depth'][a],
                                        'learning_rate': params['learning_rate'][b],
                                        'min_child_weight': params['min_child_weight'][c],
                                        'gamma': params['gamma'][d],
                                        'subsample': params['subsample'][e],
                                        'colsample_bytree': params['colsample_bytree'][f]
                                    }
                                    print_input(args_input, args_mdinput, param)
                                    model = XGBClassifier(**args_mdinput[0], **param)
                                    tStart = datetime.now()
                                    model.fit(**args_mdinput[1])
                                    tElapsed = datetime.now() - tStart
                                    f1, pr_auc, opt_thres, opt_f1 = evaluation(hyperparam, model, dataset['testX'], dataset['testY'], csv_path1, tElapsed)
                                    param['best_iteration'] = model.best_iteration
                                    search_results(f1, pr_auc, opt_thres, opt_f1, param, csv_path2)
                                    # save model
                                    if pr_auc+opt_f1 > args_input.save_model_thres:
                                        joblib.dump(model, mdl_path, compress=3, protocol=4)
    elif mdlname=='catboost':
        f_range = [f_range] if f_range is not None else range(len(params['min_data_in_leaf']))
        e_range = [e_range] if e_range is not None else range(len(params['reg_lambda']))
        d_range = [d_range] if d_range is not None else range(len(params['subsample']))
        c_range = [c_range] if c_range is not None else range(len(params['max_leaves']))
        b_range = [b_range] if b_range is not None else range(len(params['learning_rate']))
        a_range = [a_range] if a_range is not None else range(len(params['max_depth']))
        for f in f_range:
            for e in e_range:
                for d in d_range:
                    for c in c_range:
                        for b in b_range:
                            for a in a_range:
                                hyperparam = '{}_{}_{}_{}_{}_{}'.format(params['min_data_in_leaf'][f],params['reg_lambda'][e], params['subsample'][d], params['max_leaves'][c], params['learning_rate'][b], params['max_depth'][a])
                                mdl_path = os.path.join(mdl_folder, f'{db_name}-{hyperparam}.cbm')
                                if os.path.isfile(mdl_path):
                                    print('{} exist, skip current condition'.format(os.path.basename(mdl_path)))
                                else:
                                    param = {
                                        'max_depth': params['max_depth'][a],
                                        'learning_rate': params['learning_rate'][b],
                                        'max_leaves': params['max_leaves'][c],
                                        'subsample': params['subsample'][d],
                                        'reg_lambda': params['reg_lambda'][e],
                                        'min_data_in_leaf': params['min_data_in_leaf'][f]
                                    }
                                    print_input(args_input, args_mdinput, param)
                                    model = CatBoostClassifier(**args_mdinput[0], **param)
                                    tStart = datetime.now()
                                    model.fit(**args_mdinput[1])
                                    tElapsed = datetime.now() - tStart
                                    f1, pr_auc, opt_thres, opt_f1 = evaluation(hyperparam, model, dataset, csv_path1, tElapsed)
                                    param['best_iteration'] = model.best_iteration_
                                    search_results(f1, pr_auc, opt_thres, opt_f1, param, csv_path2)
                                    # save model
                                    if pr_auc+opt_f1 > args_input.save_model_thres:
                                        model.save_model(mdl_path, format="cbm")

    elif mdlname=='lightgbm':
        a = b = c = d = 0
        hyperparam = '{}_{}_{}_{}'.format(params['min_data_in_leaf'][d], params['num_leaves'][c], params['learning_rate'][b], params['max_depth'][a])
        mdl_path = os.path.join(mdl_folder, f'{db_name}-{hyperparam}.joblib')
        if not os.path.isfile(mdl_path):
            param = {
                'max_depth': params['max_depth'][a],
                'learning_rate': params['learning_rate'][b],
                'num_leaves': params['num_leaves'][c],
                'min_data_in_leaf': params['min_data_in_leaf'][d]
            }
            print_input(args_input, args_mdinput, param)
            model = LGBMClassifier(**args_mdinput[0], **param)
            tStart = datetime.now()
            model.fit(**args_mdinput[1])
            tElapsed = datetime.now() - tStart
            f1, pr_auc, opt_thres, opt_f1 = evaluation(hyperparam, model, dataset['testX'], dataset['testY'], csv_path1, tElapsed)
            search_results(f1, pr_auc, opt_thres, opt_f1, param, csv_path2)
            # save model
            if pr_auc+opt_f1 > args_input.save_model_thres:
                joblib.dump(model, mdl_path, compress=3, protocol=4)
