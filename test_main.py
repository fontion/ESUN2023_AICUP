"""
模型預測主程式
"""
# Syntax
#       python test_main.py [--model_FirstUse=json_path] [--model_UsedBefore=json_path]
import os
import joblib
import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
from catboost import Pool
from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from utils import catboost_fillna_pred
from datetime import date

def parse_args():
    """
    parse input
    """
    parser = ArgumentParser()
    # __file__ = '/root/ESUN/codes/test_main.py'
    prj_folder = os.path.dirname(__file__)
    # json_path1 = os.path.join(prj_folder,'train_test','catboost_search','eval_metric=PRAUC_use_weights','v3','embedding_5_models.json')
    json_path1 = os.path.join(prj_folder,'pretrain_models','FirstUse','embedding_3_models.json')
    json_path2 = os.path.join(prj_folder,'pretrain_models','UsedBeforeFraud','db-v4-UsedBeforeFraud-1_3.5_0.66_31_0.8_8(allin)_opt_thres.json')
    json_path3 = os.path.join(prj_folder,'pretrain_models','AfterFraud','db-v4-AfterFraud-1_3_0.8_31_0.8_8(allin)_opt_thres.json')
    mdl_path = os.path.join(prj_folder,'AF_model_final')
    parser.add_argument('--folder', type=str, choices=['dataset_1st','dataset_2nd','dataset_3rd'], default='dataset_3rd')
    parser.add_argument('--model_FirstUse', type=str, default=json_path1)
    parser.add_argument('--model_UsedBeforeFraud', type=str, default=json_path2)
    parser.add_argument('--model_AfterFraud', type=str, default=json_path3)
    parser.add_argument('--model_AfterFraud2', type=str, default=mdl_path)
    args = parser.parse_args()
    return args, prj_folder


if __name__=='__main__':
    args, prj_folder = parse_args()

    db_folder = os.path.join(prj_folder, args.folder)
    if args.folder=='dataset_1st':
        path_pred_new = os.path.join(db_folder, 'public_add_features.joblib')
    elif args.folder=='dataset_2nd':
        path_pred_new = os.path.join(db_folder, 'private_add_features.joblib')
    elif args.folder=='dataset_3rd':
        path_pred_new = os.path.join(db_folder, 'private2_add_features.joblib')
    if not os.path.isfile(path_pred_new):
        raise AssertionError('add_features file not found, please run create_testset.py first')

    df_pred_new = joblib.load(path_pred_new)
    catboost_fillna_pred(df_pred_new)
    lg1 = (df_pred_new.Pusage==0) | (df_pred_new.Cusage==0)
    lg2 = (df_pred_new.Pusage > 0) & (df_pred_new.Cusage > 0)
    lg2 &= (df_pred_new.Plabel_incidence==0) & (df_pred_new.Clabel_incidence==0)
    lg3 = (df_pred_new.Plabel_incidence > 0) | (df_pred_new.Clabel_incidence > 0)
    lg1[lg1&lg3] = False # 第一和第三段可能有交集，交給第3段模型做
    # assert lg1.equals(~lg2), 'unexpect error'

    # mdl_folder = os.path.join(prj_folder, 'train_test', 'catboost_search_可刪')
    mdl_folder1 = os.path.dirname(args.model_FirstUse)
    mdl_folder2 = os.path.dirname(args.model_UsedBeforeFraud)
    mdl_folder3 = os.path.dirname(args.model_AfterFraud)
    with open(args.model_FirstUse,'rt') as f:
        info1 = json.load(f)
    with open(args.model_UsedBeforeFraud, 'rt') as f:
        info2 = json.load(f)
    with open(args.model_AfterFraud, 'rt') as f:
        info3 = json.load(f)

    assert info1['mdlname']=='catboost' and info2['mdlname']=='catboost' and info3['mdlname']=='catboost', 'Only support catboost model currently.'
    models_name1 = info1['mdlfile']
    models_name2 = info2['mdlfile']
    models_name3 = info3['mdlfile']
    # models_name1 = [
    #     'db-v2-FirstUse-2_0.8_31_0.04_8(allin).cbm',
    #     'db-v2-FirstUse-3_0.66_31_0.04_6(allin).cbm',
    #     'db-v2-FirstUse-3_0.75_31_0.03_8(allin).cbm'
    # ]
    # models_name2 = [
    #     'db-v2-UsedBefore-2_0.7_31_0.03_8(allin).cbm',
    #     'db-v2-UsedBefore-3_0.66_31_0.02_8(allin).cbm',
    #     'db-v2-UsedBefore-3_0.66_31_0.03_8(allin).cbm'
    # ]
    X = df_pred_new
    cat_features = np.nonzero(((X.dtypes==bool)|(X.dtypes=='category')).to_numpy())[0]

    # model baseline
    model = CatBoostClassifier()
    model.load_model(os.path.join(mdl_folder3, models_name3[0]))
    baseline = model.predict(X.loc[lg3], prediction_type='RawFormulaVal')

    pool1 = Pool(X.loc[lg1], cat_features=cat_features)
    pool2 = Pool(X.loc[lg2], cat_features=cat_features)
    pool3 = Pool(X.loc[lg3], cat_features=cat_features, baseline=baseline)
    n_sample1 = lg1.sum()
    n_sample2 = lg2.sum()
    n_sample3 = lg3.sum()
    y_score1 = np.zeros(n_sample1)
    y_score2 = np.zeros(n_sample2)
    # y_score3 = np.zeros(n_sample3)
    for m in range(len(models_name1)):
        model = CatBoostClassifier()
        model.load_model(os.path.join(mdl_folder1, models_name1[m]))
        y_score1 += model.predict_proba(pool1)[:,1]
    for m in range(len(models_name2)):
        model = CatBoostClassifier()
        model.load_model(os.path.join(mdl_folder2, models_name2[m]))
        y_score2 += model.predict_proba(pool2)[:,1]
    # for m in range(len(models_name3)):
    model = CatBoostClassifier()
    model.load_model(args.model_AfterFraud2)
    y_score3 = model.predict_proba(pool3)[:,1]

    opt_thres_emb1 = info1['opt_thres']
    opt_thres_emb2 = info2['opt_thres']
    opt_thres_emb3 = info3['opt_thres']
    y_pred = np.zeros(X.shape[0], dtype=np.uint8)
    y_pred[lg1.to_numpy()] = (y_score1 > opt_thres_emb1).astype(np.uint8)
    y_pred[lg2.to_numpy()] = (y_score2 > opt_thres_emb2).astype(np.uint8)
    y_pred[lg3.to_numpy()] = (y_score3 > opt_thres_emb2).astype(np.uint8)

    # csv_path = os.path.join(prj_folder, '31_範例繳交檔案.csv')
    csv_path = os.path.join(prj_folder, 'dataset_3rd', 'private_2_template_v2.csv')
    df = pd.read_csv(csv_path)
    df.set_index('txkey', inplace=True)
    df.loc[X.index,'pred'] = y_pred
    df.reset_index(inplace=True)

    # prepare output
    submit_folder = os.path.join(prj_folder, 'submit')
    if not os.path.isdir(submit_folder):
        os.mkdir(submit_folder)
    daystr = date.today().strftime('%m%d')
    csv_path = os.path.join(prj_folder, 'submit', 'private_2(TEAM_4334).csv')
    df.to_csv(csv_path, index=False)