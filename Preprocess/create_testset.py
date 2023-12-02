"""
產生正式預測時public或private set時所需的所有features
"""
# Syntax
#       python create_testset.py ['dataset_1st'/'dataset_2nd']
import os
import joblib
import pandas as pd
from argparse import ArgumentParser
from preprocess import format_dtypeI
from create_dataset import add_features
from create_dataset import renew_chid_cano
from create_dataset import union_chid_cano_categories
from datetime import datetime

def get_df_pred_new(path_train, path_pred):
    """
    產生新的features矩陣包含所有features供模型預測
    """
    # load raw data
    if isinstance(path_train,str):
        df_train = joblib.load(path_train)
    elif isinstance(path_train, list):
        df_train = pd.concat([joblib.load(p) for p in path_train], axis=0, ignore_index=True)
    df_pred = joblib.load(path_pred)
    format_dtypeI(df_train, df_pred)

    df_train = df_train.set_index('txkey').drop_duplicates()
    df_pred = df_pred.set_index('txkey') # 需保留所有txkey，不能drop_duplicates
    assert df_train.columns[:-1].equals(df_pred.columns), 'features mismatch'
    assert df_train.index.is_unique, 'txkey is not unique'
    assert df_pred.index.is_unique, 'txkey is not unique'
    # add days_from_start
    df_train.insert(0,'days_from_start',df_train.locdt + df_train.loctm/86400) # 取代授權日期locdt和授權時間loctm
    df_pred.insert(0,'days_from_start',df_pred.locdt + df_pred.loctm/86400) # 取代授權日期locdt和授權時間loctm
    df_train.sort_values(['locdt','chid','loctm'], inplace=True)
    df_pred.sort_values(['locdt','chid','loctm'], inplace=True)
    # replace loctm as category data type
    categories = ['00-03','03-06','06-09','09-12','12-15','15-18','18-21','21-24'] # 每三小時為一個類別
    df_train.loctm = pd.Categorical.from_codes(df_train.loctm//(3*60*60), categories=categories, ordered=True)
    df_pred.loctm = pd.Categorical.from_codes(df_pred.loctm//(3*60*60), categories=categories, ordered=True)
    # add discount
    df_train.insert(df_train.shape[1],'discount', df_train.conam-df_train.flam1)
    df_pred.insert(df_pred.shape[1],'discount', df_pred.conam-df_pred.flam1)
    # add new category 'clear' in chid and cano which represent no fraud before
    df_train.chid = df_train.chid.cat.add_categories('clear')
    df_train.cano = df_train.cano.cat.add_categories('clear')
    df_pred.chid = df_pred.chid.cat.add_categories('clear')
    df_pred.cano = df_pred.cano.cat.add_categories('clear')

    # add group features
    drop_cols = ['locdt','flam1']
    df_pred_add = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame(),parallel=True)[0]
    df_pred.drop(columns=drop_cols, inplace=True)
    df_pred_new = pd.concat([df_pred.loc[df_pred_add.index], df_pred_add], axis=1)
    renew_chid_cano(df_pred_new)
    return df_pred_new

def add_categories(path_db, df_pred_add):
    """
    讓categorical features類別一致 (相同categorical feature, testing set需包含所有training set的類別)
    """
    D = joblib.load(path_db)
    union_chid_cano_categories(D['pred'], df_pred_add)

if __name__=='__main__':
    tStart = datetime.now()
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, choices=['dataset_1st','dataset_2nd','dataset_3rd'], default='dataset_3rd')
    args = parser.parse_args()

    # __file__=='/root/ESUN/codes/create_testset.py'
    prj_folder = os.path.dirname(os.path.dirname(__file__))
    db_folder = os.path.join(prj_folder, args.folder)
    if args.folder=='dataset_1st':
        path_train = os.path.join(db_folder, 'training_raw.joblib')
        path_pred = os.path.join(db_folder, 'public_processed_raw.joblib')
        path_db = os.path.join(db_folder, 'db-v3.joblib')
        path_pred_new = os.path.join(db_folder, 'public_add_features.joblib')
    elif args.folder=='dataset_2nd':
        path_train = [
            os.path.join(prj_folder, 'dataset_1st', 'training_raw.joblib'),
            os.path.join(prj_folder, 'dataset_2nd', 'public_raw.joblib'),
        ]
        path_pred = os.path.join(db_folder, 'private_1_processed_raw.joblib')
        path_db = os.path.join(db_folder, 'db-v4.joblib')
        path_pred_new = os.path.join(db_folder, 'private_add_features.joblib')
    elif args.folder=='dataset_3rd':
        path_train = [
            os.path.join(prj_folder, 'dataset_1st', 'training_raw.joblib'),
            os.path.join(prj_folder, 'dataset_2nd', 'public_raw.joblib'),
            os.path.join(prj_folder, 'dataset_3rd', 'private_1_raw.joblib')
        ]
        path_pred = os.path.join(db_folder, 'private_2_processed_raw.joblib')
        path_db = os.path.join(db_folder, 'db-v5.joblib')
        path_pred_new = os.path.join(db_folder, 'private2_add_features.joblib')

    df_pred_new = get_df_pred_new(path_train, path_pred)
    add_categories(path_db, df_pred_new)
    joblib.dump(df_pred_new, path_pred_new, compress=3, protocol=4)
    print('Elapsed time:', datetime.now() - tStart)