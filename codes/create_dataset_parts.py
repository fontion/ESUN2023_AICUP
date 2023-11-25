# Syntax
#       python create_dataset.py [df_pred/day1/day2/.../day54]
import os
import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime
from preprocess import format_dtypeI
from create_dataset import unique_vec
from create_dataset import add_features
from create_dataset import add_features_default
from create_dataset import renew_chid_cano
from create_dataset import assign_chid_cano_categories
from create_dataset import union_chid_cano_categories

if __name__=='__main__':
    # __file__ = '/root/ESUN/ceodes/create_dataset.py'
    tStart = datetime.now()
    db_folder1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_1st')
    db_folder2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_2nd') # TODO: need modify
    mode = 'combine_training_and_public' # [split_from_training/combine_training_and_public]
    part = 'df_pred'
    if len(sys.argv) > 1:
        part = sys.argv[1]

    print('Create mode:', mode)
    print('Process part:', part)

    if mode=='split_from_training':
        # df_raw = pd.read_csv(os.path.join(db_folder1, 'training.csv'))
        df_raw = joblib.load(os.path.join(db_folder1, 'training_raw.joblib'))
        lg = df_raw.locdt > 51 # 取出52, 53, 54, 55四天的記錄做為testing set，佔原來training data的7.19%
        df_train = df_raw.loc[~lg].copy()
        df_pred = df_raw.loc[lg].copy() # for validation
        df_raw = '' # release memory occupied by df_raw
        format_dtypeI(df_train, df_pred)
        db_output = db_folder1
        period = 4 # 一次預測的週期是幾天(public set是4天)
    elif mode=='combine_training_and_public':
        path_train = os.path.join(db_folder1,'training_raw.joblib')
        path_public = os.path.join(db_folder2,'public_raw.joblib')
        df_raw = pd.concat([joblib.load(path_train), joblib.load(path_public)], axis=0)
        lg = df_raw.locdt > 54 # 取出55, 56, 57, 58, 59五天的記錄做為testing set，佔原來training data的8.24%
        df_train = df_raw.loc[~lg].copy()
        df_pred = df_raw.loc[lg].copy() # for validation
        df_raw = '' # release memory occupied by df_raw
        format_dtypeI(df_train, df_pred)
        db_output = db_folder2
        period = 5 # 一次預測的週期是幾天(private set是5天) 複賽時視情況調整
    else:
        raise AssertionError(f'Unexpect mode: {mode}')

    db_temp = os.path.join(db_output,'temp')
    if not os.path.isdir(db_temp):
        os.mkdir(db_temp)
    df_train = df_train.set_index('txkey').drop_duplicates()
    df_pred = df_pred.set_index('txkey').drop_duplicates()
    assert df_train.columns.equals(df_pred.columns), 'features mismatch'
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

    drop_cols = ['locdt','flam1']
    # deal with df_pred
    path_pred = os.path.join(db_temp,'db_pred.joblib')
    if os.path.isfile(path_pred):
        print('[Loading df_pred_add from file...]')
        df_pred_new = joblib.load(path_pred)
    else:
        if part=='df_pred':
            print('[Deal with df_pred]')
            df_pred_add, lg_in = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame())
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
                df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame())
                sli_pred = np.r_[sli_pred][lg_in.to_numpy()]
                # data augmentation
                for j in range(1,period):
                    if i > j:
                        sli_train = slice(0, unqix[i-j])
                        df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), df_add)
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
        path_db = os.path.join(db_output,'db-v4.joblib')
        joblib.dump({'train':df_train_new, 'pred':df_pred_new}, path_db, compress=3, protocol=4)

        assert df_pred_new.shape[0]==df_pred_new.drop_duplicates().shape[0], 'Found duplicate records'
        assert df_train_new.shape[0]==df_train_new.drop_duplicates().shape[0], 'Found duplicate records'

    tElapsed = datetime.now() - tStart
    print('Elapsed time:', tElapsed)