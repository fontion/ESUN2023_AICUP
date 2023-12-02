# Syntax
#       python create_dataset_incremental.py --mode=split_from_training --period=4
import os
import pandas as pd
import numpy as np
import joblib
import concurrent.futures
from argparse import ArgumentParser
from datetime import datetime
from preprocess import format_dtypeI
from create_dataset import format_dtypeII
from create_dataset import unique_vec
from create_dataset import add_features
from create_dataset import renew_chid_cano
from create_dataset import assign_chid_cano_categories
from create_dataset import union_chid_cano_categories

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['split_from_training','combine_training_and_public','final_competition'], default='final_competition')
    parser.add_argument('--db_name', type=str, default='db-v5')
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.add_argument('--max_workers', type=int, default=None)
    args, extra = parser.parse_known_args()
    parser = ArgumentParser()
    if args.mode=='split_from_training':
        parser.add_argument('--period', type=int, default=4) # 一次預測的週期是幾天(public set是4天)
    elif args.mode=='combine_training_and_public':
        parser.add_argument('--period', type=int, default=5) # 一次預測的週期是幾天(private set是5天)
    elif args.mode=='final_competition':
        parser.add_argument('--period', type=int, default=5) # 一次預測的週期是幾天(private set2是5天) 複賽
    args = parser.parse_args(extra, args)
    return args

def load_joblib(db_temp, i):
    return joblib.load(os.path.join(db_temp, f'db_day{i:02}.joblib'))

def combine_all_parts(df_train, db_temp, args):
    path_pred = os.path.join(db_temp,'db_pred.joblib')
    print('[Loading df_pred_add from file...]')
    df_pred_new = joblib.load(path_pred)

    locdt = df_train.locdt.to_numpy()
    unqix = unique_vec(locdt)
    assert unqix.size-1==locdt[-1]+1, 'Not everyday has records'
    df_train_new = []
    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor(args.max_workers) as executor:
            df_train_new = executor.map(
                load_joblib,
                [db_temp]*(locdt[-1]+1),
                range(locdt[-1],-1,-1)
            )
        df_train_new = list(df_train_new)
    else:
        for i in range(locdt[-1],-1,-1): # 到第1天
            path_train = os.path.join(db_temp, f'db_day{i:02}.joblib')
            print(f'[Load df_train-day{i:02} from file...]')
            df_train_new.append(joblib.load(path_train))
    
    assign_chid_cano_categories(df_train_new)
    df_train_new = pd.concat(df_train_new, axis=0)
    
    # df_pred_new需包含所有df_train_new的類別，所以從df_train_new的categories再去新增
    union_chid_cano_categories(df_train_new, df_pred_new)
    return df_train_new, df_pred_new

def update_categories(df_train, df_daily):
    catcols = df_daily.columns[df_daily.dtypes=='category'].tolist()
    catcols.remove('chid')
    catcols.remove('cano')
    for col in catcols:
        if col[0] in 'PC':
            dtype1 = df_train[col.split('_')[0][1:]].dtype
        else:
            dtype1 = df_train[col].dtype
        dtype2 = df_daily[col].dtype
        if dtype1!=dtype2:
            assert df_daily.loc[df_daily[col].notna(),col].isin(dtype1.categories).all(), 'categories not contains all values, please check'
            df_daily[col] = df_daily[col].astype(dtype1)

def add_features_single_day(df_train, db_pre, db_temp, unqix, i, j):
    print('[Deal with df_train]',f'day-{i:02}')
    path_train_src = os.path.join(db_pre, f'db_day{i:02}.joblib')
    path_train_dst = os.path.join(db_temp, f'db_day{i:02}.joblib')
    sli_pred = slice(unqix[i], unqix[i+1])
    sli_train = slice(0, unqix[i-j])
    df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame(), parallel=False)
    df_daily_new = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
    renew_chid_cano(df_daily_new)
    df_daily_pre = joblib.load(path_train_src)
    update_categories(df_train, df_daily_pre)
    df_train_daily = [df_daily_new, df_daily_pre]
    assign_chid_cano_categories(df_train_daily)
    df_train_daily = pd.concat(df_train_daily, axis=0)
    joblib.dump(df_train_daily.drop_duplicates(), path_train_dst, compress=3, protocol=4)

def load_and_update(df_train, path_train_src, path_train_dst):
    df_daily = joblib.load(path_train_src)
    update_categories(df_train, df_daily)
    joblib.dump(df_daily, path_train_dst, compress=3, protocol=4)

if __name__=='__main__':
    args = parse_args()

    __file__ = '/root/ESUN/Preprocess/create_dataset_incremental.py'
    tStart = datetime.now()
    db_folder1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_1st')
    db_folder2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_2nd')
    print('Create mode:', args.mode)

    path_train = os.path.join(db_folder1,'training_raw.joblib')
    path_public = os.path.join(db_folder2,'public_raw.joblib')
    if args.mode=='split_from_training':
        df_raw = joblib.load(path_train)
        db_output = db_folder1
        # 取出52, 53, 54, 55四天的記錄做為testing set，佔原來training data的7.19%
    elif args.mode=='combine_training_and_public':
        df_raw = pd.concat([joblib.load(path_train), joblib.load(path_public)], axis=0, ignore_index=True)
        db_output = db_folder2
        db_output_pre = db_folder1
        # 取出55, 56, 57, 58, 59五天的記錄做為testing set，佔原來training data的8.24%
    
    lg = df_raw.locdt > (df_raw.locdt.max()-args.period)
    df_train = df_raw.loc[~lg].copy()
    df_pred = df_raw.loc[lg].copy() # for validation
    df_raw = '' # release memory occupied by df_raw
    format_dtypeI(df_train, df_pred)
    df_train, df_pred = format_dtypeII(df_train, df_pred)
    db_temp = os.path.join(db_output,f'predict_{args.period}_days')
    drop_cols = ['locdt','flam1']
    if not os.path.isdir(db_temp):
        os.mkdir(db_temp)
        db_pre = os.path.join(db_output,f'predict_{args.period-1}_days-gt')
        if not os.path.isdir(db_pre):
            db_pre = os.path.join(db_output,f'predict_{args.period-1}_days')
        if os.path.isdir(db_pre): # 同dataset資料夾，補上缺少的一天
            # deal with pred
            path_pred = os.path.join(db_temp,'db_pred.joblib')
            print('[Deal with df_pred]')
            df_pred_add, lg_in = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame(),parallel=True)
            df_pred.drop(columns=drop_cols, inplace=True)
            df_pred_new = pd.concat([df_pred.loc[df_pred_add.index], df_pred_add], axis=1)
            renew_chid_cano(df_pred_new)
            joblib.dump(df_pred_new, path_pred, compress=3, protocol=4)
            # deal with df_train
            locdt = df_train.locdt.to_numpy()
            unqix = unique_vec(locdt)
            assert unqix.size-1==locdt[-1]+1, 'Not everyday has records'
            j = args.period-1
            if args.parallel:
                # 使用dataset_1st, period=5做測試，max_workers=unlimit, 耗時 0:26:12.398744
                # 使用dataset_1st, period=5做測試, max_workers=16, 耗時 0:29:31.596493

                n_task = locdt[-1] - j
                with concurrent.futures.ProcessPoolExecutor(args.max_workers) as executor:
                    executor.map(
                        add_features_single_day,
                        [df_train]*n_task,
                        [db_pre]*n_task,
                        [db_temp]*n_task,
                        [unqix]*n_task,
                        range(locdt[-1], j, -1),
                        [j]*n_task
                    )
                with concurrent.futures.ProcessPoolExecutor(args.max_workers) as executor:
                    executor.map(
                        load_and_update, # (df_train, path_train_src, path_train_dst)
                        [df_train.iloc[:0]]*n_task,
                        (os.path.join(db_pre, f'db_day{i:02}.joblib') for i in range(j,-1,-1)),
                        (os.path.join(db_temp, f'db_day{i:02}.joblib') for i in range(j,-1,-1))
                    )
            else:
                # 使用dataset_1st, period=5做測試，耗時 1:06:46.396630
                for i in range(locdt[-1], j, -1):
                    print('[Deal with df_train]',f'day-{i:02}')
                    path_train_src = os.path.join(db_pre, f'db_day{i:02}.joblib')
                    path_train_dst = os.path.join(db_temp, f'db_day{i:02}.joblib')
                    sli_pred = slice(unqix[i], unqix[i+1])
                    sli_train = slice(0, unqix[i-j])
                    df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame(), parallel=True)
                    df_daily_new = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
                    renew_chid_cano(df_daily_new)
                    df_daily_pre = joblib.load(path_train_src)
                    update_categories(df_train, df_daily_pre)
                    df_train_daily = [df_daily_new, df_daily_pre]
                    assign_chid_cano_categories(df_train_daily)
                    df_train_daily = pd.concat(df_train_daily, axis=0)
                    joblib.dump(df_train_daily.drop_duplicates(), path_train_dst, compress=3, protocol=4)
                for i in range(j,-1,-1):
                    print('[Deal with df_train]',f'day-{i:02}')
                    path_train_src = os.path.join(db_pre, f'db_day{i:02}.joblib')
                    path_train_dst = os.path.join(db_temp, f'db_day{i:02}.joblib')
                    load_and_update(df_train, path_train_src, path_train_dst)
        else:
            raise AssertionError(f"previous result not found: {db_pre}")

    df_train_new, df_pred_new = combine_all_parts(df_train, db_temp, args)
    path_db = os.path.join(db_output,f'{args.db_name}.joblib')
    joblib.dump({'train':df_train_new, 'pred':df_pred_new}, path_db, compress=3, protocol=4)
    
    tElapsed = datetime.now() - tStart
    print('Elapsed time:', tElapsed)

    assert df_pred_new.shape[0]==df_pred_new.drop_duplicates().shape[0], 'Found duplicate records'
    assert df_train_new.shape[0]==df_train_new.drop_duplicates().shape[0], 'Found duplicate records'
