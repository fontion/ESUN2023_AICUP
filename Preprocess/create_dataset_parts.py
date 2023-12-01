"""
將create_dataset.py工作拆分, 以方便使用shell多線程處理, 縮短處理時間
"""
# Syntax
#       python create_dataset.py [df_pred/day1/day2/.../day54]
import os
import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime
from preprocess import format_dtypeI
from create_dataset import format_dtypeII
from create_dataset import unique_vec
from create_dataset import add_features
from create_dataset import add_features_default
from create_dataset import renew_chid_cano
from create_dataset import assign_chid_cano_categories
from create_dataset import union_chid_cano_categories

if __name__=='__main__':
    # __file__ = '/root/ESUN/Preprocess/create_dataset_parts.py'
    tStart = datetime.now()
    db_folder1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_1st')
    db_folder2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_2nd')
    mode = 'combine_training_and_public' # [split_from_training/combine_training_and_public]
    part = 'df_pred'
    if len(sys.argv) > 1:
        part = sys.argv[1]

    print('Create mode:', mode)
    print('Process part:', part)

    path_train = os.path.join(db_folder1,'training_raw.joblib')
    path_public = os.path.join(db_folder2,'public_raw.joblib')
    if mode=='split_from_training':
        df_raw = joblib.load(path_train)
        db_output = db_folder1
        # 取出52, 53, 54, 55四天的記錄做為testing set，佔原來training data的7.19%
        period = 4 # 一次預測的週期是幾天(public set是4天)
    elif mode=='combine_training_and_public':
        df_raw = pd.concat([joblib.load(path_train), joblib.load(path_public)], axis=0)
        db_output = db_folder2
        # 取出55, 56, 57, 58, 59五天的記錄做為testing set，佔原來training data的8.24%
        period = 5 # 一次預測的週期是幾天(private set是5天) 複賽時視情況調整
    else:
        raise AssertionError(f'Unexpect mode: {mode}')

    lg = df_raw.locdt > (df_raw.locdt.max()-period)
    df_train = df_raw.loc[~lg].copy()
    df_pred = df_raw.loc[lg].copy() # for validation
    df_raw = '' # release memory occupied by df_raw
    format_dtypeI(df_train, df_pred)
    df_train, df_pred = format_dtypeII(df_train, df_pred)
    
    db_temp = os.path.join(db_output,f'predict_{period}_days')
    if not os.path.isdir(db_temp):
        os.mkdir(db_temp)

    drop_cols = ['locdt','flam1']
    # deal with df_pred
    path_pred = os.path.join(db_temp,'db_pred.joblib')
    if os.path.isfile(path_pred):
        print('[Loading df_pred_add from file...]')
        df_pred_new = joblib.load(path_pred)
    else:
        if part=='df_pred':
            print('[Deal with df_pred]')
            df_pred_add, lg_in = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame(),parallel=True)
            df_pred.drop(columns=drop_cols, inplace=True)
            df_pred_new = pd.concat([df_pred.loc[df_pred_add.index], df_pred_add], axis=1)
            renew_chid_cano(df_pred_new)
            joblib.dump(df_pred_new, path_pred, compress=3, protocol=4)
        else:
            df_pred_new = None

    # deal with df_train
    locdt = df_train.locdt.to_numpy()
    unqix = unique_vec(locdt)
    assert unqix.size-1==locdt[-1]+1, 'Not everyday has records'
    df_train_new = []
    for i in range(locdt[-1],0,-1): # 到第1天
        path_train = os.path.join(db_temp, f'db_day{i:02}.joblib')
        if os.path.isfile(path_train):
            print(f'[Load df_train-day{i:02} from file...]')
            df_train_new.append(joblib.load(path_train))
        else:
            if f'day{i}'==part:
                print('[Deal with df_train]',f'day-{i:02}')
                sli_pred = slice(unqix[i], unqix[i+1])
                sli_train = slice(0, unqix[i])
                df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame(), parallel=True)
                sli_pred = np.r_[sli_pred][lg_in.to_numpy()]
                # data augmentation
                for j in range(1,period):
                    if i > j:
                        sli_train = slice(0, unqix[i-j])
                        df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), df_add, parallel=True)
                        sli_pred = sli_pred[lg_in.to_numpy()]
                        if sli_pred.size==0: break
                df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
                renew_chid_cano(df_train_daily)
                df_train_new.append(df_train_daily.drop_duplicates())
                joblib.dump(df_train_new[-1], path_train, compress=3, protocol=4)

    path_train = os.path.join(db_temp, f'db_day00.joblib')
    if os.path.isfile(path_train):
        print('[Load df_train-day00 from file...]')
        df_train_new.append(joblib.load(path_train))
    else:
        if part=='day0':
            print('[Deal with df_train] day0')
            sli_pred = slice(0, unqix[1])
            df_add = add_features_default(df_train.iloc[sli_pred], pd.DataFrame(), 'PC')
            df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
            renew_chid_cano(df_train_daily)
            df_train_new.append(df_train_daily.drop_duplicates())
            joblib.dump(df_train_new[-1], path_train, compress=3, protocol=4)
            

    if len(df_train_new)==unqix.size-1 and isinstance(df_pred_new, pd.DataFrame):
        assign_chid_cano_categories(df_train_new)
        df_train_new = pd.concat(df_train_new, axis=0)

        # df_pred_new需包含所有df_train_new的類別，所以從df_train_new的categories再去新增
        union_chid_cano_categories(df_train_new, df_pred_new)

        # output database
        path_db = os.path.join(db_output,'db-v4-p7d.joblib')
        joblib.dump({'train':df_train_new, 'pred':df_pred_new}, path_db, compress=3, protocol=4)

        assert df_pred_new.shape[0]==df_pred_new.drop_duplicates().shape[0], 'Found duplicate records'
        assert df_train_new.shape[0]==df_train_new.drop_duplicates().shape[0], 'Found duplicate records'

    tElapsed = datetime.now() - tStart
    print('Elapsed time:', tElapsed)