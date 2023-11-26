"""
計算模型判斷positive或negative最佳的threshold
"""

# Syntax
#       python opt_thres_final.py [db_name] [mdlname] [param1] [param2] ... [--subfolder]

import json
import joblib
import os
import numpy as np
from argparse import ArgumentParser
from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from utils import generate_dic
from sklearn.metrics import precision_recall_curve

def parse_args():
    """
    parse input
    """
    parser = ArgumentParser()
    parser.add_argument('db_name', type=str)
    parser.add_argument('mdlname', type=str)
    parser.add_argument('params', nargs='+')
    args, extra = parser.parse_known_args()
    if args.mdlname=='catboost':
        if 'FirstUse' in args.db_name:
            # subfolder = os.path.join('eval_metric=PRAUC_use_weights',args.db_name[3:5])
            subfolder = os.path.join('eval_metric=F1', args.db_name[3:5])
        elif 'UsedBefore' in args.db_name:
            subfolder = os.path.join('eval_metric=F1', args.db_name[3:5])
    else:
        subfolder = ''
    parser = ArgumentParser()
    parser.add_argument('--subfolder', type=str, default=subfolder)
    parser.parse_args(extra, args)
    return args

if __name__=='__main__':
    args = parse_args()
    print('Database:', args.db_name)
    print('Model:', args.mdlname)
    print('Embedding model parameters:')
    print(json.dumps(args.params, indent=4))
    # __file__ = '/root/ESUN/codes/opt_thres_final.py'


    prj_folder = os.path.dirname(os.path.dirname(__file__))
    if int(args.db_name[4]) < 4:
        path_db = os.path.join(prj_folder,'dataset_1st',f'{args.db_name}.joblib')
    else:
        path_db = os.path.join(prj_folder,'dataset_2nd',f'{args.db_name}.joblib')
    dataset = generate_dic(path_db, args.mdlname, allin=True)
    mdl_folder = os.path.join(prj_folder,'train_test',f'{args.mdlname}_search', args.subfolder)
    y_score = np.zeros(dataset['trainY'].size)
    mdlfile = []
    for p in args.params:
        if args.mdlname=='catboost':
            mdlfile.append(f'{args.db_name}-{p}(allin).cbm')
            mdl_path = os.path.join(mdl_folder, mdlfile[-1])
            model = CatBoostClassifier()
            model.load_model(mdl_path)
        elif args.mdlname=='xgboost':
            mdlfile.append(f'{args.db_name}-{p}(allin).joblib')
            mdl_path = os.path.join(mdl_folder, mdlfile[-1])
            model = joblib.load(mdl_path)
        y_score += model.predict_proba(dataset['trainX'])[:,1]

    # select optimal threshold
    precision, recall, threshold = precision_recall_curve(dataset['trainY'].to_numpy(), y_score)
    ix = np.nanargmax(2*precision*recall/(precision+recall))
    opt_thres = threshold[ix]

    n_model = len(args.params)
    json_path = os.path.join(mdl_folder, f'embedding_{n_model}_models.json')
    info = {
        'db_name': args.db_name,
        'mdlname': args.mdlname,
        'mdlfile': mdlfile,
        'opt_thres': opt_thres
    }
    with open(json_path, 'wt') as f:
        json.dump(info, f, indent=4)